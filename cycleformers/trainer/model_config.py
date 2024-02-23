from dataclasses import dataclass, field
import math
from typing import Optional, Union, Dict
from yaml import safe_dump, safe_load

from peft import PeftConfig
from transformers import SchedulerType, TrainingArguments
from transformers.training_args import OptimizerNames


@dataclass
class ModelConfig:
    '''
    Will pull in unset values from TrainingArguments object from the calling trainer
    '''
    per_device_train_batch_size: int = field(
        default=None,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=None,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for training."
            )
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated, the use of `--per_device_eval_batch_size` is preferred. "
                "Batch size per GPU/TPU core/CPU for evaluation."
            )
        },
    )
    gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )
    eval_delay: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " evaluation_strategy."
            )
        },
    )
    learning_rate: float = field(
        default=None,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=None,
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=None,
        metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=None,
        metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=None,
        metadata={"help": "Epsilon for AdamW optimizer."}
    )
    max_grad_norm: float = field(
        default=None,
        metadata={"help": "Max gradient norm."}
    )
    num_train_epochs: float = field(
        default=None,
        metadata={"help": "Total number of training epochs to perform."}
    )
    max_steps: int = field(
        default=None,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default=None,
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_kwargs: Optional[Dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts"
            )
        },
    )
    warmup_ratio: float = field(
        default=None,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(
        default=None,
        metadata={"help": "Linear warmup over warmup_steps."}
    )
    gradient_checkpointing: bool = field(
        default=None,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    optim: Union[OptimizerNames, str] = field(
        default=None,
        metadata={"help": "The optimizer to use."},
    )
    optim_args: Optional[str] = field(
        default=None,
        metadata={"help": "Optional arguments to supply to optimizer."},
    )
    

    def fill_from_training_args(self, training_args: TrainingArguments):
        '''
        Fills in any unset values from the `training_args` object. This is allows global training
        arguments to be set and then specific model training arguments to be set on top of them.
        '''
        if not isinstance(training_args, TrainingArguments):
            raise ValueError(f'`training_args` must be an instance of `TrainingArguments` class. Got {type(training_args)} instead.')
        
        for key, value in self.__dict__.items():
            if value is None:
                setattr(self, key, getattr(training_args, key))


    # def __post_init__(self):
    #     if self.peft_config is not None and not isinstance(self.peft_config, PeftConfig):
    #         raise ValueError(f'`peft_config` must be an instance of `PeftConfig` class. Got {type(self.peft_config)} instead.')
    
    # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
    def get_warmup_steps(self, num_training_steps: int) -> int:
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps
        

    def to_dict(self):
        return self.__dict__
    
    def to_yaml(self, file_path: str, args_key: str = None) -> None:
        '''
        Save the model configuration to a YAML file. The model configuration can
        be saved under s specific key in the YAML file. To specify the key, use
        the `args_key` parameter. If no key is specified, the model configuration
        will be saved at the root of the YAML file.

        Args:
            file_path (str): The path to the YAML file.
            args_key (str): The key to use to save the model configuration to the
                YAML file.
        '''
        with open(file_path, 'w') as file:
            safe_dump({args_key: self.to_dict()}, file, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, dictionary: dict) -> 'ModelConfig':
        return cls(**dictionary)
    
    @classmethod
    def from_yaml(cls, file_path: str, args_key: str = 'model') -> 'ModelConfig':
        '''
        Load a model configuration from a YAML file. Accepts configuration of a
        model from YAML file. The model configuration can be in its own YAML or
        as part of a larger configuration file. To specify the model
        configuration in a larger configuration file, use the `args_key`
        parameter to specify the key.

        Args:
            file_path (str): The path to the YAML file.
            args_key (str): The key to use to load the model configuration from
                the YAML file.

        Returns:
            ModelConfig: The model configuration.
        '''
        with open(file_path, 'r') as file:
            config = safe_load(file)

        if args_key in config:
            return cls(**config['model'])
        else:
            return cls(**config)