from typing import Callable, Dict, List, Tuple, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.utils import logging
from transformers.utils.generic import can_return_loss
from transformers.utils.import_utils import is_peft_available

from .training_args import ModelTrainingArguments

if is_peft_available():
    from peft import PeftModel


logger = logging.get_logger(__name__)


class _ModelHandler:
    '''
    Class to hold model, tokenizer, optimizer and scheduler and model specific training arguments
    in one place to make it easier to pass around. This class is not meant to be used directly.
    '''
    create_optimiser_and_scheduler = Trainer.create_optimizer_and_scheduler
    create_accelerator_and_postprocess = Trainer.create_accelerator_and_postprocess
    is_local_process_zero = Trainer.is_local_process_zero
    is_world_process_zero = Trainer.is_world_process_zero
    _move_model_to_device = Trainer._move_model_to_device

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, 'PeftModel'],
        args: ModelTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        optimizers: Optional[Tuple[Optimizer, LambdaLR]],
        callbacks: Optional[List[TrainerCallback]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        self.args = args
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.create_accelerator_and_postprocess()

        args._setup_devices

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )
        
        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        if getattr(model, "hf_device_map", None) is not None:
            devices = [device for device in set(model.hf_device_map.values()) if device not in ["cpu", "disk"]]
            if len(devices) > 1:
                self.is_model_parallel = True
            elif len(devices) == 1:
                self.is_model_parallel = self.args.device != torch.device(devices[0])
            else:
                self.is_model_parallel = False

            # warn users
            if self.is_model_parallel:
                logger.info(
                    "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                    " to `True` to avoid any unexpected behavior such as device placement mismatching."
                )
                logger.warning(
                    'Parallel models are untested in this version oc CycleFormers. Please report any issues you encounter.'
                )

        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or self.is_deepspeed_enabled
            # TODO: Add other conditions as they are implemented
            or self.is_fsdp_enabled
        ):
            self.place_model_on_device = False

        self.tokenizer = tokenizer

        if self.place_model_on_device:
            self._move_model_to_device(model, args.device)
        
        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers

        if (self.is_deepspeed_enabled or self.is_fsdp_enabled) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if Deepspeed or PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )

        self._loggers_enabled = False
        
        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if (args.fp16 or args.bf16) and args.half_precision_backend == "auto":
            if args.device == torch.device("cpu"):
                if args.fp16:
                    raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                else:
                    args.half_precision_backend = "cpu_amp"
            logger.info(f"Using {args.half_precision_backend} half precision backend")

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )

        self.control = TrainerControl()
        self.current_flos = 0
        self.label_names = 'labels'
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False