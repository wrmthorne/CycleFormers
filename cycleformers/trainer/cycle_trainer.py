import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
    EvalPrediction,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import deepspeed_init
from transformers.trainer_callback import (
    CallbackHandler,
    PrinterCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    enable_full_determinism,
    set_seed,
    TrainerMemoryTracker,
)
from transformers.utils import logging
from transformers.utils.import_utils import (
    is_in_notebook,
    is_peft_available,
    is_sagemaker_mp_enabled
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

if is_peft_available():
    from peft import PeftModel


logger = logging.get_logger(__name__)


class CycleTrainer(Trainer):
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
                ).update_from_global_args(args)),
                tokenizers[name],
                model_init=model_init.get(name, None),
                compute_metrics=compute_metrics.get(name, None),
                callbacks=callbacks.get(name, None),
                optimizers=optimizers.get(name, (None, None)),
                preprocess_logits_for_metrics=preprocess_logits_for_metrics.get(name, None),
            ) for name in self._model_names
        }

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

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            num_examples = self.num_examples(train_dataloader)
        
        for handler in self.handlers.values():
            handler.delay_optimizer_creation = is_sagemaker_mp_enabled() or handler.is_fsdp_xla_enabled or handler.is_fsdp_enabled

            if handler._created_lr_scheduler:
                handler.lr_scheduler = None
                handler._created_lr_scheduler = False

            if handler.is_deepspeed_enabled:
                handler.optimizer, handler.lr_scheduler = deepspeed_init(handler, num_training_steps=handler.args.max_steps)

            if not handler.delay_optimizer_creation:
                handler.create_optimizer_and_scheduler(num_training_steps=handler.args.max_steps)

        self.state = TrainerState() # TODO: How to handle state for multiple sub models?

        for handler in self.handlers.values():
            if handler.delay_optimizer_creation:
                handler.create_optimizer_and_scheduler(num_training_steps=handler.args.max_steps)

        # Train!
        logger.info('***** Running training *****')
        logger.info(f'  Num examples = {num_examples:,}')

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        for handler in self.handlers.values():
            handler.callback_handler.model = handler.model
            handler.callback_handler.optimizer = handler.optimizer
            handler.callback_handler.lr_scheduler = handler.lr_scheduler
            handler.callback_handler.train_dataloader = train_dataloader

            handler.state.max_steps = handler.args.max_steps
            handler.state.num_train_epochs = handler.args.num_train_epochs
            handler.state.is_local_process_zero = handler.is_local_process_zero()
            handler.state.is_world_process_zero = handler.is_world_process_zero()

            handler.tr_loss = torch.tensor(0.0).to(handler.args.device)
            handler._total_loss_scalar = 0.0
            handler._globalstep_last_logged = handler.state.global_step

            handler.model.zero_grad()

            handler.control = handler.callback_handler.on_train_begin(handler.args, handler.state, handler.control)

        self._globalstep_last_logged = self.state.global_step

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        total_batched_samples = 0
        # TODO: Handle variable number of epochs between models
        for epoch in range(int(max(handler.args.num_train_epochs for handler in self.handlers.values()))):
            epoch_iterator = train_dataloader

            steps_in_epoch = len(epoch_iterator)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                for name, handler in self.handlers.items():
                    if step % handler.args.gradient_accumulation_steps == 0:
                        handler.control = handler.callback_handler.on_step_begin(handler.args, handler.state, handler.control)

                    if inputs[name]:
                        with handler.accelerator.accumulate(handler.model):
                            handler.tr_loss_step = self.training_step(handler, inputs[name], self.handlers)
                            print(f'Model {name} loss: {handler.tr_loss_step}')

                    if total_batched_samples % handler.args.gradient_accumulation_steps == 0:
                        handler.optimizer.step()
                        handler.optimizer_was_run = not handler.accelerator.optimizer_step_was_skipped
                        if handler.optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(handler.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                handler.lr_scheduler.step()

                        handler.model.zero_grad()
                        handler.control = handler.callback_handler.on_step_end(handler.args, handler.state, handler.control)
                        print(f'Optimised model: {name}')
                    else:
                        handler.control = handler.callback_handler.on_substep_end(handler.args, handler.state, handler.control)
                        
                        

        
        metrics = {}
        self._memory_tracker.stop_and_update_metrics(metrics)


    def training_step(self, curr_handler, inputs, all_handlers):
        '''
        Basic training step with fully regularly trains all models in the trainer.

        Subclass and override this method to implement custom training steps.
        '''
        model = curr_handler.model

        model.train()
        inputs = curr_handler._prepare_inputs(inputs)

        with curr_handler.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            loss.backward()

        if curr_handler.args.n_gpu > 1:
            loss = loss.mean()

        return loss.detach() / curr_handler.args.gradient_accumulation_steps
    

    def log(self, logs):
        if self.global_state.epoch is not None:
            logs['epoch'] = round(self.global_state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs['num_input_tokens_seen'] = self.global_state.num_input_tokens

        output = {**logs, **{"step": self.global_state.global_step}}

        