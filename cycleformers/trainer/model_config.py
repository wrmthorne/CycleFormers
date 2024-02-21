from dataclasses import dataclass, field
import math
from typing import Optional
from yaml import safe_dump, safe_load

from peft import PeftConfig
from transformers import GenerationConfig

PRETRAINED_MODEL_KWARGS = [
    'pretrained_model_name_or_path',
    'torch_dtype',
    'trust_remote_code',
    'use_flash_attention_2',
]

PRETRAINED_TOKENIZER_KWARGS = [
    'pretrained_model_name_or_path',
    'use_fast',
    'padding_side',
    'padding',
    'truncation',
]

TRAINING_KWARGS = [
    'per_device_train_batch_size',
    'per_device_eval_batch_size',
    'learning_rate',
    'weight_decay',
    'adam_beta1',
    'adam_beta2',
    'adam_epsilon',
    'lr_scheduler_type',
    'lr_scheduler_kwargs',
    'warmup_ratio',
    'warmup_steps',
    'optim',
    'optim_args',
    'resume_from_checkpoint',
    'hub_token',
]


@dataclass
class ModelConfig:
    '''
    Configuration class for each model in the cycle
    '''
    # Model specific
    pretrained_model_name_or_path: Optional[str] = field(
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
    use_flash_attention_2: bool = field(
        default=False,
        metadata={'help': 'Use Flash attention for the model. This is a separate package that must be installed from `flash-attn`.'}
    )

    # PEFT specific
    peft_config: Optional[PeftConfig] = field(
        default=None,
        metadata={'help': 'PEFT configuration for the model. Leave None to not use PEFT.'}
    )

    # Generation specific
    generation_config: Optional[GenerationConfig] = field(
        default=GenerationConfig(),
        metadata={'help': 'The generation configuration for the model.'}
    )

    # Training specific
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

    # Tokenizer specific
    padding_side: str = field(
        default='right',
        metadata={'help': 'The side to pad on.'}
    )
    use_fast: bool = field(
        default=True,
        metadata={'help': 'Use fast tokenizers.'}
    )
    padding: str = field(
        default=False,
        metadata={'help': 'The padding strategy.'}
    )
    truncation: str = field(
        default=False,
        metadata={'help': 'The truncation strategy.'}
    )

    def __post_init__(self):
        if self.peft_config is not None and not isinstance(self.peft_config, PeftConfig):
            raise ValueError(f'`peft_config` must be an instance of `PeftConfig` class. Got {type(self.peft_config)} instead.')
    
    def pretrained_model_kwargs(self):
        return {k: v for k, v in self.to_dict().items() if k in PRETRAINED_MODEL_KWARGS}
    
    def pretrained_tokenizer_kwargs(self):
        return {k: v for k, v in self.to_dict().items() if k in PRETRAINED_TOKENIZER_KWARGS}
    
    def training_kwargs(self):
        return {k: v for k, v in self.to_dict().items() if k in TRAINING_KWARGS}
    
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