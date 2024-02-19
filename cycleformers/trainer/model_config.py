from dataclasses import dataclass, field
from peft import PeftConfig
from transformers.training_args import OptimizerNames
from transformers import Trainer
from typing import List, Optional

@dataclass
class ModelConfig:
    '''
    Configuration class for each model in the cycle
    '''
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.'}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            'help': (
                'Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the '
                'dtype will be automatically derived from the model\'s weights.'
            ),
            'choices': ['auto', 'bfloat16', 'float16', 'float32'],
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={'help': 'Trust remote code when loading a model.'}
    )
    peft_config: Optional[PeftConfig] = field(
        default=None,
        metadata={'help': 'PEFT configuration for the model. Leave None to not use PEFT.'}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={'help': 'Use Flash attention for the model. This is a separate package that must be installed from `flash-attn`.'}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={'help': 'The batch size per GPU for training.'}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={'help': 'The batch size per GPU for evaluation.'}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={'help': 'The learning rate for the model.'}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={'help': 'The weight decay for the model.'}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={'help': 'The beta1 value for the Adam optimizer.'}
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={'help': 'The beta2 value for the Adam optimizer.'}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={'help': 'The epsilon value for the Adam optimizer.'}
    )
    lr_scheduler_type: str = field(
        default='linear',
        metadata={'help': 'The learning rate scheduler type.'}
    )
    lr_scheduler_kwargs: Optional[dict] = field(
        default=None,
        metadata={'help': 'The learning rate scheduler kwargs.'}
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={'help': 'The warmup ratio for the learning rate scheduler.'}
    )
    warmup_steps: int = field(
        default=0,
        metadata={'help': 'The warmup steps for the learning rate scheduler.'}
    )
    optim: str = field(
        default='adamw_torch',
        metadata={'help': 'The optimizer to use.'}
    )
    optim_args: Optional[str] = field(
        default=None,
        metadata={'help': 'The optimizer kwargs.'}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={'help': 'The path to a checkpoint from which to resume training.'}
    )
    hub_token: Optional[str] = field(
        default=None,
        metadata={'help': 'The Hugging Face model hub token.'}
    )

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value

        return output_dict
    
    def __post_init__(self):
        if self.peft_config is not None and not isinstance(self.peft_config, PeftConfig):
            raise ValueError(f'`peft_config` must be an instance of `PeftConfig` class. Got {type(self.peft_config)} instead.')
        
    def get_optimiser(self):
        return Trainer.get_optimizer_cls_and_kwargs(self)