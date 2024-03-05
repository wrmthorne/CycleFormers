import math
import os
import sys
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from datasets import Dataset
import numpy as np
from packaging import version
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DataCollator,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    default_data_collator,
)
from transformers.integrations import deepspeed_init
from transformers.trainer_callback import (
    CallbackHandler,
    PrinterCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
)
from transformers.trainer_utils import (
    TrainerMemoryTracker,
    enable_full_determinism,
    set_seed,
    speed_metrics,
)
from transformers.utils import logging
from transformers.utils.import_utils import (
    is_accelerate_available,
    is_apex_available,
    is_in_notebook,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

from cycleformers.data import CombinedLoader
from .training_args import ModelTrainingArguments, TrainingArguments
from .model_handler import _ModelHandler
from .trainer_utils import (
    has_length,
)

DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_accelerate_available():
    from accelerate.utils import (
        DistributedType,
    )

if is_apex_available():
    from apex import amp

if is_peft_available():
    from peft import PeftModel

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


logger = logging.get_logger(__name__)


class MultiModelTrainer(Trainer):
    '''
    CycleTrainer is a wrapper class for the Huggingface Trainer class, specifically designed to
    train models using cycle consistency training but aiming to be a generic multi-model trainer.

    Args:
        models (`dict` or `PeftModel`, *required*):
            The models to be trained. If a dictionary is supplied, the keys should be the names of the
            models and the values should be the models themselves. If a PeftModel is supplied, there must
            be at least two attached adapters and the model will be trained using those adapters.
        tokenizers (`dict` or `PreTrainedTokenizerBase`, *required*):
            The tokenizers to be used for each model. If a dictionary is supplied, the keys should be the
            names of the models and the values should be the tokenizers themselves. If a single tokenizer
            is supplied, it will be used for all models.
        args (`TrainerConfig`, *optional*):
            General, non model-specific training arguments. If not supplied, default arguments will be used.
        model_args (`dict`, *optional*):
            The model arguments to be used for training each model. The keys should be the names of the models
            and the values should be the model arguments themselves. If a single set of model arguments is
            supplied, it will be used for all models. If no model arguments are supplied for a specific model,
            the default model arguments will be used.
        data_collators (`dict` or `DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of the dataset. If a dictionary is
            supplied, the keys should be the names of the models and the values should be the data collators
            themselves. If a single data collator is supplied, it will be used for all models. If no collator
            is supplied for a specific model, a collator will be chosen based on the model type.
        train_datasets (`dict`, *optional*):
            The training datasets to be used for each model. The keys should be the names of the models and
            the values should be the datasets themselves.
        eval_datasets (`dict`, *optional*):
            The evaluation datasets to be used for each model. The keys should be the names of the models and
            the values should be the datasets themselves. A subset of models can be evaluated by supplying
            only a subset of model names.
        optimizers (`dict`, *optional*):
            The optimizers to be used for each model. The keys should be the names of the models and the
            values should be a tuple of the optimizer and the learning rate scheduler. Will default to
            AdamW and a linear learning rate scheduler if nothing is supplied for any specific model.
    '''
    # TODO: Fix type hinting
    def __init__(
        self,
        models: Union[Dict[str, Union[PreTrainedModel, nn.Module]], 'PeftModel'],
        tokenizers: Union[Dict[str, PreTrainedTokenizerBase], PreTrainedTokenizerBase],
        args: Optional[TrainingArguments] = None,
        model_args: Optional[Union[Dict[str, ModelTrainingArguments], ModelTrainingArguments]] = dict(),
        data_collators: Optional[Union[Dict[str, DataCollator], DataCollator]] = dict(),
        train_datasets: Optional[Dict[str, Dataset]] = dict(),
        eval_datasets: Optional[Dict[str, Dataset]] = None,
        model_init: Optional[Dict[str, Callable[[], PreTrainedModel]]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[Union[Dict[str, List[TrainerCallback]], List[TrainerCallback]]] = None,
        optimizers: Optional[Dict[str, Tuple[Optimizer, LambdaLR]]] = dict(),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        # TODO: Load Trainer args as default and overwrite them with model args if present. Warn if overwriting
        if args is None:
            output_dir = 'tmp_trainer'
            logger.info(f'No `TrainingArguments` passed, using `output_dir={output_dir}`.')
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the models when using models
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.is_in_train = False

        # ================
        # TEMPORARY UNTIL Union TYPES ARE HANDLED
        if model_init is None:
            model_init = {}
        if compute_metrics is None:
            compute_metrics = {}
        if callbacks is None:
            callbacks = {}
        if preprocess_logits_for_metrics is None:
            preprocess_logits_for_metrics = {}
        # ================

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # TODO: Handle case where any arg is not a dict
        self._model_names = list(models.keys())

        self.handlers = {
            name: _ModelHandler(
                models[name],
                model_args.get(name, ModelTrainingArguments(
                    output_dir=os.path.join(args.output_dir, name)
                )).update_from_global_args(args),
                tokenizers[name],
                model_init=model_init.get(name, None),
                compute_metrics=compute_metrics.get(name, None),
                callbacks=callbacks.get(name, None),
                optimizers=optimizers.get(name, (None, None)),
                preprocess_logits_for_metrics=preprocess_logits_for_metrics.get(name, None),
            ) for name in self._model_names
        }

        for name, handler in self.handlers.items():
            handler._name = name

        # Create collators for any datasets that don't have one based on the model with the same name
        for name in train_datasets:
            collator = data_collators.get(name, None)
            if collator is None:
                if self.handlers[name].tokenizer is None:
                    data_collators[name] = default_data_collator
                else:
                    data_collators[name] = DataCollatorWithPadding(self.handlers[name].tokenizer)
            else:
                if not callable(collator) and callable(getattr(collator, "collate_batch", None)):
                    raise ValueError(
                        f'Invalid data collator for model {name}. Must be a callable and have a '
                         'collate_batch method.'
                    )
                
        if any(name not in self._model_names for name in data_collators):
            raise ValueError(
                f'Unrecognised model names in data_collators. Expected {self._model_names}, '
                f'got {list(data_collators.keys())}.'
            )
        
        self.data_collators = data_collators

        for name, dataset in train_datasets.items():
            if dataset is not None and not has_length(dataset) and self.handlers[name].args.max_steps <= 0:
                raise ValueError(
                    f"train_dataset {name} does not implement __len__, max_steps has to be specified. "
                    "The number of steps needs to be known in advance for the learning rate scheduler."
                )

            if (
                dataset is not None
                and isinstance(dataset, torch.utils.data.IterableDataset)
                and self.handlers[name].args.group_by_length
            ):
                raise ValueError(
                    f"train_dataset {name} uses the `--group_by_length` option but this is only available "
                    "for `Dataset`, not `IterableDataset"
                )
        
        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets

        self.compute_metrics = compute_metrics
        callbacks = []#[handler.callback_handler for handler in self.handlers.values()]
        self.callback_handler = CallbackHandler(
            callbacks, None, None, None, None
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        self.label_smoother = None

        self.state = TrainerState()
        self.control = TrainerControl()

        self.current_flos = 0
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)      

        self._memory_tracker.stop_and_update_metrics()
            

    def get_train_dataloader(self) -> DataLoader:
        '''
        Method to create the training dataloader for the models.
        
        Returns:
            CombinedLoader: The combined dataloader for the all models.
        '''
        if self.train_datasets is None:
            raise ValueError('No training datasets found. Please supply training datasets.')
        
        if any(name not in self._model_names for name in self.train_datasets):
            raise ValueError(
                f'Unrecognised model names in train_datasets. Expected {self._model_names}, '
                f'got {list(self.train_datasets.keys())}.'
            )
        
        data_loaders = {}
        for name, dataset in self.train_datasets.items():
            data_loaders[name] = DataLoader(
                dataset,
                batch_size=self.handlers[name].args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collators[name]
            )

        return CombinedLoader(data_loaders, 'max_size')
    

    def get_eval_dataloader(self, eval_datasets: Optional[Dict[str, Dataset]] = None) -> DataLoader:
        '''
        Method to create the evaluation dataloader for the models.
        
        Args:
            eval_datasets (`dict`, *optional*):
                The evaluation datasets to be used for each model. The keys should be the names of the models and
                the values should be the datasets themselves. A subset of models can be evaluated by supplying
                only a subset of model names.
                
        Returns:
            CombinedLoader: The combined dataloader for the all models.
        '''
        if (len(eval_datasets) == 0 or eval_datasets is None) and \
           (len(self.eval_datasets) == 0 or self.eval_datasets is None):
            raise ValueError('No evaluation datasets found. Please supply evaluation datasets.')
        
        eval_datasets = eval_datasets if eval_datasets is not None else self.eval_datasets

        if any(name not in self._model_names for name in eval_datasets):
            raise ValueError(
                f'Unrecognised model names in eval_datasets. Expected {self._model_names}, '
                f'got {list(eval_datasets.keys())}.'
            )

        data_loaders = {}
        for name, dataset in eval_datasets.items():
            data_loaders[name] = DataLoader(
                dataset,
                batch_size=self.handlers[name].args.per_device_eval_batch_size,
                shuffle=True,
                collate_fn=self.data_collators[name]
            )

        return CombinedLoader(data_loaders, 'max_size')

    def train(self):
        '''
        Main training entry point.

        MORE COMPATABILITY STUFF WILL BE ADDED HERE EVENTUALLY
        '''
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        args = self.args

        self.is_in_train = True

        for handler in self.handlers.values():
            if handler.args.neftune_noise_alpha is not None:
                raise NotImplementedError('Neftune noise is not yet implemented for CycleTrainer.')
            
            if (handler.args.fp16_full_eval or handler.args.bf16_full_eval) and not handler.args.do_train:
                handler._move_model_to_device(handler.model, handler.args.device)

            # TODO: Finish later
                
        inner_training_loop = self._inner_training_loop
        return inner_training_loop(args=args)
    
    def _inner_training_loop(self, args=None):
        '''
        Inner training loop for the CycleTrainer.

        MORE COMPATABILITY STUFF WILL BE ADDED HERE EVENTUALLY
        '''
        for handler in self.handlers.values():
            handler.accelerator.free_memory()
        
        # logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        train_dataloader = iter(self.get_train_dataloader()) # TODO: Fix this stupid need to call iter

        len_dataloader = len(train_dataloader) if has_length(train_dataloader) else None
        num_train_tokens = None

        for name, handler in self.handlers.items():
            handler.grad_norm = None

            handler.total_train_batch_size = handler.args.per_device_train_batch_size * handler.args.gradient_accumulation_steps * handler.args.world_size

            # TODO: Properly handle max_steps calculation per dataset
            if len_dataloader is not None:
                handler.num_update_steps_per_epoch = len_dataloader // handler.args.gradient_accumulation_steps
                handler.num_update_steps_per_epoch = max(handler.num_update_steps_per_epoch, 1)

                if handler.args.max_steps > 0:
                    handler.max_steps = handler.args.max_steps
                    handler.num_train_epochs = handler.args.max_steps // handler.num_update_steps_per_epoch + int(
                        args.max_steps % handler.num_update_steps_per_epoch > 0
                    )
                    handler.num_train_samples = handler.max_steps * handler.total_train_batch_size
                    # TODO: Implement tokens per second calculation

                else:
                    handler.max_steps = math.ceil(handler.args.num_train_epochs * handler.num_update_steps_per_epoch)
                    handler.num_train_epochs = math.ceil(handler.args.num_train_epochs)
                    handler.num_train_samples = self.num_examples(train_dataloader) * handler.args.num_train_epochs
            elif handler.args.max_steps > 0:
                handler.max_steps = handler.args.max_steps
                handler.num_train_epochs = sys.maxsize
                handler.num_updates_per_epoch = handler.max_steps
                handler.num_train_samples = handler.max_steps * handler.total_train_batch_size
                # TODO: Implement tokens per second calculation
            else:
                raise ValueError(
                    "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                    f" {handler.args.max_steps} for model {name}"
                )
            
            handler.delay_optimizer_creation = is_sagemaker_mp_enabled() or handler.is_fsdp_xla_enabled or handler.is_fsdp_enabled

            if handler._created_lr_scheduler:
                handler.lr_scheduler = None
                handler._created_lr_scheduler = False

            if handler.is_deepspeed_enabled:
                handler.optimizer, handler.lr_scheduler = deepspeed_init(handler, num_training_steps=handler.max_steps)

            if not handler.delay_optimizer_creation:
                handler.create_optimizer_and_scheduler(num_training_steps=handler.max_steps)

            handler.state = TrainerState()

            handler.state.train_batch_size = handler.args.per_device_train_batch_size

            handler.state.max_steps = handler.max_steps
            handler.state.num_train_epochs = handler.num_train_epochs

            if handler.args.logging_steps is not None:
                if handler.args.logging_steps < 1:
                    handler.state.logging_steps = math.ceil(handler.max_steps * handler.args.logging_steps)
                else:
                    handler.state.logging_steps = handler.args.logging_steps
            if handler.args.eval_steps is not None:
                if handler.args.eval_steps < 1:
                    handler.state.eval_steps = math.ceil(handler.max_steps * handler.args.eval_steps)
                else:
                    handler.state.eval_steps = handler.args.eval_steps
            if handler.args.save_steps is not None:
                if handler.args.save_steps < 1:
                    handler.state.save_steps = math.ceil(handler.max_steps * handler.args.save_steps)
                else:
                    handler.state.save_steps = handler.args.save_steps

            # Activate gradient checkpointing if needed
            if handler.args.gradient_checkpointing:
                if handler.args.gradient_checkpointing_kwargs is None:
                    handler.gradient_checkpointing_kwargs = {}
                else:
                    handler.gradient_checkpointing_kwargs = handler.args.gradient_checkpointing_kwargs

                handler.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=handler.gradient_checkpointing_kwargs)

            handler._model = handler._wrap_model(handler.model_wrapped)

            handler.use_accelerator_prepare = True if handler._model is handler.model else False

            if handler.delay_optimizer_creation:
                if handler.use_accelerator_prepare:
                    handler.model = handler.accelerator.prepare(handler.model)
                handler.create_optimizer_and_scheduler(num_training_steps=handler.max_steps)

            if handler.use_accelerator_prepare:
                handler.model.train()
                if hasattr(handler.lr_scheduler, "step"):
                    if handler.use_apex:
                        handler._model = handler.accelerator.prepare(self.model)
                    else:
                        handler._model, handler.optimizer = handler.accelerator.prepare(handler.model, handler.optimizer)
                else:
                    # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                    handler._model, handler.optimizer, handler.lr_scheduler = handler.accelerator.prepare(
                        handler.model, handler.optimizer, handler.lr_scheduler
                    )

            if handler.is_fsdp_enabled:
                handler.model = handler.model_wrapped = handler._model

            if handler._model is not handler.model:
                handler.model_wrapped = handler._model

            if handler.is_deepspeed_enabled:
                handler.deepspeed = handler.model_wrapped

            handler.state.epoch = 0

            handler.callback_handler.model = handler._model
            handler.callback_handler.optimizer = handler.optimizer
            handler.callback_handler.lr_scheduler = handler.lr_scheduler
            handler.callback_handler.train_dataloader = train_dataloader

            handler.state.is_local_process_zero = handler.is_local_process_zero()
            handler.state.is_world_process_zero = handler.is_world_process_zero()

            handler.tr_loss = torch.tensor(0.0).to(handler.args.device)
            handler._total_loss_scalar = 0.0
            handler._globalstep_last_logged = handler.state.global_step

            handler.model.zero_grad()

            handler.control = handler.callback_handler.on_train_begin(handler.args, handler.state, handler.control)

        # Train!
        logger.info('***** Running training *****')
        # logger.info(f'  Num examples = {num_examples:,}')

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        self._globalstep_last_logged = self.state.global_step

        total_batched_samples = 0
        # TODO: Handle variable number of epochs between models
        for epoch in range(int(max(handler.args.num_train_epochs for handler in self.handlers.values()))):
            epoch_iterator = train_dataloader

            for handler in self.handlers.values():
                handler.control = handler.callback_handler.on_epoch_begin(handler.args, handler.state, handler.control)

                # TODO: Update to respoect varying sizes of datasets
                handler.steps_in_epoch = (
                    len(epoch_iterator)
                    if len_dataloader is not None
                    else self.args.max_steps * self.args.gradient_accumulation_steps
                )

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                for name, handler in self.handlers.items():
                    is_last_step_and_steps_less_than_grad_acc = (
                        handler.steps_in_epoch <= handler.args.gradient_accumulation_steps and (step + 1) == handler.steps_in_epoch
                    )
                    
                    if step % handler.args.gradient_accumulation_steps == 0:
                        handler.control = handler.callback_handler.on_step_begin(handler.args, handler.state, handler.control)

                    if inputs[name]:
                        with handler.accelerator.accumulate(handler._model):
                            handler.tr_loss_step = self.training_step(handler, inputs[name], self.handlers)

                    if (
                        handler.args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and torch.isnan(handler.tr_loss_step) or torch.isinf(handler.tr_loss_step)
                    ):
                        handler.tr_loss = handler.tr_loss / (1 + handler.state.global_step - handler._globalstep_last_logged)
                    else:
                        handler.tr_loss += handler.tr_loss_step

                    handler.current_flos += float(handler.floating_point_ops(inputs[name]))

                    is_last_step_and_steps_less_than_grad_acc = (
                        handler.steps_in_epoch <= handler.args.gradient_accumulation_steps and (step + 1) == handler.steps_in_epoch
                    )

                    if (
                        total_batched_samples % handler.args.gradient_accumulation_steps == 0
                        or is_last_step_and_steps_less_than_grad_acc
                    ):
                        if is_last_step_and_steps_less_than_grad_acc:
                            handler.accelerator.gradient_state._set_sync_gradients(True)

                        if handler.args.max_grad_norm is not None and handler.args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and handler.args.fp16:
                                _grad_norm = handler.optimizer.clip_master_grads(handler.args.max_grad_norm)
                            elif handler.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(handler.optimizer),
                                    handler.args.max_grad_norm,
                                )
                            else:
                                _grad_norm = handler.accelerator.clip_grad_norm_(
                                    handler.model.parameters(),
                                    handler.args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and handler.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                handler.grad_norm = handler._model.get_global_grad_norm()
                            else:
                                handler.grad_norm = _grad_norm.item() if _grad_norm is not None else None

                        handler.optimizer.step()
                        handler.optimizer_was_run = not handler.accelerator.optimizer_step_was_skipped
                        if handler.optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(handler.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                handler.lr_scheduler.step()

                        handler._model.zero_grad()
                        handler.state.global_step += 1
                        handler.state.epoch = epoch + (step + 1) / handler.steps_in_epoch
                        handler.control = handler.callback_handler.on_step_end(handler.args, handler.state, handler.control)

                        # TODO: Need to fix trial and ignore_keys_for_eval
                        handler._maybe_log_save_evaluate(handler.tr_loss, handler.grad_norm, handler._model, None, epoch, None)
                    else:
                        handler.control = handler.callback_handler.on_substep_end(handler.args, handler.state, handler.control)
                        
                for handler in self.handlers.values():
                    handler.control = handler.callback_handler.on_epoch_end(handler.args, handler.state, handler.control)
                    handler._maybe_log_save_evaluate(handler.tr_loss, handler.grad_norm, handler._model, None, epoch, None)
                    
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        self.is_in_train = False
        self._memory_tracker.stop()

        for handler in self.handlers.values():
            handler._total_loss_scalar += handler.tr_loss.item()
            handler.train_loss = handler._total_loss_scalar / handler.state.global_step

            handler.metrics = speed_metrics(
                'train',
                start_time,
                num_samples=handler.num_train_samples,
                num_steps=handler.state.max_steps,
                num_tokens=handler.num_train_tokens,
            )
            handler.store_flos()
            handler.metrics['total_flos'] = handler.state.total_flos
            handler.metrics['train_loss'] = handler.train_loss

            self._memory_tracker.update_metrics(handler.metrics)

            handler.log(handler.metrics)

    def training_step(self, curr_handler, inputs, all_handlers):
        '''
        Basic training step with fully regularly trains all models in the trainer.

        Subclass and override this method to implement custom training steps.
        '''
        model = curr_handler._model

        model.train()
        inputs = curr_handler._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with curr_handler.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            loss.backward()

        if curr_handler.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / curr_handler.args.gradient_accumulation_steps
    

    def log(self, logs):
        if self.global_state.epoch is not None:
            logs['epoch'] = round(self.global_state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs['num_input_tokens_seen'] = self.global_state.num_input_tokens

        output = {**logs, **{"step": self.global_state.global_step}}

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ):
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description='Evaluation',
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = 'eval'
    ):
        
        for handler in self.handlers.values():
            handler._model = handler._wrap_model(handler.model, training=False, dataloader=dataloader)

            if len(handler.accelerator._models) == 0 and handler._model is handler.model:
                handler._model = (
                    handler.accelerator.prepare(handler._model)
                    if handler.is_deepseed_enabled
                    else handler.accelerator.prepare_model(handler._model, evaluation_mode=True)
                )

                if handler.is_fsdp_enabled:
                    handler.model = handler._model

                if handler._model is not handler.model:
                    handler.model_wrapped = handler._model

                if handler.is_deepspeed_enabled:
                    handler.deepspeed = handler.model_wrapped
        
            if not self.is_in_train:
                if handler.fp16_full_eval:
                    handler._model = handler._model.to(dtype=torch.float16, device=handler.args.device)
                elif handler.bf16_full_eval:
                    handler._model = handler._model.to(dtype=torch.bfloat16, device=handler.args.device)

            handler._model.eval()

            handler.callback_handler.eval_dataloader = dataloader

        logger.info(f'***** Running {description} *****')
        if has_length(dataloader):
            logger.info(f'  Num examples = {self.num_examples(dataloader)}')
        else:
            logger.info('  Num examples: Unknown')

        for step, inputs in enumerate(dataloader):
            for name, handler in self.handlers.items():
                if name not in inputs:
                    continue

                handler.observed_batch_size = find_batch_size(inputs[name])
                if handler.observed_batch_size is not None:
                    handler.observed_num_examples += handler.observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if handler.batch_size is None:
                        handler.batch_size = handler.observed_batch_size
                else:
                    handler.batch_size = handler.args.per_device_eval_batch_size

                loss, logits, labels = self.prediction_step(handler, inputs[name], prediction_loss_only, ignore_keys=ignore_keys)
                main_input_name = getattr(handler.model, "main_input_name", "input_ids")
                inputs_decode = handler._prepare_input(inputs[name][main_input_name]) if handler.args.include_inputs_for_metrics else None

                if loss is not None:
                    losses = handler.gather_function(loss.repeat(handler.batch_size))
                    handler.losses_host = losses if handler.losses_host is None else nested_concat(handler.losses_host, losses, padding_index=-100)
                if labels is not None:
                    labels = handler.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                if inputs_decode is not None:
                    inputs_decode = handler.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                    inputs_decode = handler.gather_function((inputs_decode))
                    handler.inputs_host = (
                        inputs_decode
                        if handler.inputs_host is None
                        else nested_concat(handler.inputs_host, inputs_decode, padding_index=-100)
                    )
                if logits is not None:
                    logits = handler.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                    if handler.preprocess_logits_for_metrics is not None:
                        logits = handler.preprocess_logits_for_metrics(logits, labels)
                    logits = handler.gather_function((logits))
                    handler.preds_host = logits if handler.preds_host is None else nested_concat(handler.preds_host, logits, padding_index=-100)

                if labels is not None:
                    labels = handler.gather_function((labels))
                    handler.labels_host = labels if handler.labels_host is None else nested_concat(handler.labels_host, labels, padding_index=-100)

                handler.control = handler.callback_handler.on_prediction_step_end(handler.args, handler.state, handler.control)

                if handler.args.eval_accumulation_steps is not None and (step + 1) % handler.args.eval_accumulation_steps == 0:
                    if handler.losses_host is not None:
                        losses = nested_numpify(handler.losses_host)
                        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                    if handler.preds_host is not None:
                        logits = nested_numpify(handler.preds_host)
                        all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                    if handler.inputs_host is not None:
                        inputs_decode = nested_numpify(handler.inputs_host)
                        all_inputs = (
                            inputs_decode
                            if all_inputs is None
                            else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                        )
                    if handler.labels_host is not None:
                        labels = nested_numpify(handler.labels_host)
                        all_labels = (
                            labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                        )

                    # Set back to None to begin a new accumulation
                    handler.losses_host, handler.preds_host, handler.inputs_host, handler.labels_host = None, None, None, None

    def prediction_step(
        self,
        curr_handler: _ModelHandler,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        # TODO: Update to be the same as the original trainer later
        has_labels = 'labels' in inputs
        inputs = curr_handler._prepare_inputs(inputs)

        if has_labels:
            labels = nested_detach(tuple(inputs.get('labels')))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                with curr_handler.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(curr_handler._model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ['loss'])
                else:
                    logits = outputs[1:]

            else:
                loss = None
                with curr_handler.compute_loss_context_manager():
                    outputs = curr_handler._model(**inputs)
                
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs

        if prediction_loss_only:
            return (loss, None, None)
        
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)