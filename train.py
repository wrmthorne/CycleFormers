import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from CycleModel import CycleModel
from transformers import HfArgumentParser, GenerationConfig
from dataclasses import dataclass, field
from typing import Optional, List
import yaml
import os

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='gpt2',
        metadata={"help": "Model name or path to model for model A (default: gpt2)"}
    )
    task: str = field(
        default='causal_lm',
        metadata={"help": "Task for both models (default: causal_lm)"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use bfloat16 (default: False)"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use float16 (default: False)"}
    )

@dataclass
class DataArguments:
    data_a: str = field(
        default='data/example/A',
        metadata={"help": "Path to data for model A. If this a validation set can be found, it will be loaded (default: data/example/A)"}
    )
    data_b: str = field(
        default='data/example/B',
        metadata={"help": "Path to data for model B. If this a validation set can be found, it will be loaded (default: data/example/B)"}
    )

@dataclass
class TrainingArguments:
    full_finetune: bool = field(
        default=True,
        metadata={"help": "Finetune the entire model without adapters (default: True)"}
    )
    lr_a: float = field(
        default=2e-4,
        metadata={"help": "Learning rate for model A (default: 2e-4)"}
    )
    lr_b: float = field(
        default=2e-4,
        metadata={"help": "Learning rate for model B (default: 2e-4)"}
    )
    train_batch_size: int = field(
        default=2,
        metadata={"help": "Train batch size (default: 2)"}
    )
    eval_batch_size: int = field(
        default=2,
        metadata={"help": "Eval batch size (default: 2)"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation (default: 1)"}
    )
    output_dir: str = field(
        default='models/example',
        metadata={"help": "Path to save model checkpoints (default: models/example)"}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "Quantise model to n bits (default: 16)"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension (default: 64)"}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha (default: 16)"}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout (default: 0.0)"}
    )
    modules: List[str] = field(
        default=None,
        metadata={"help": "Which modules to add LoRA modules to (default: All linear)"}
    )
    lora_bias: str = field(
        default=None,
        metadata={"help": 'Whether to use bias in the lora layers (default: None)'}
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": 'Use gradient checkpointing. You want to use this.'}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed for reproducibility (default: 42)"}
    )

@dataclass
class GenerationArguments:
    max_length: int = field(
        default=128,
        metadata={"help": "Max length for generation (default: 128)"}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate (default: None)"}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True) 

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0) 
    repetition_penalty: Optional[float] = field(default=1.0) 
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

@dataclass
class LightningArguments:
    log_every_n_steps: int = field(
        default=1,
        metadata={"help": "Log every n steps (default: 1)"}
    )
    val_check_interval: float = field(
        default=1.0,
        metadata={"help": "Validate every n epochs (default: 1.0)"}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use wandb for logging (default: False)"}
    )
    num_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs to train for (default: 1)"}
    )
    wandb_project: str = field(
        default='cycle-lightning',
        metadata={"help": "Wandb project name (default: cycle-lightning)"}
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "Wandb entity name (default: None)"}
    )
    wandb_run_name: str = field(
        default='cycle-lightning',
        metadata={"help": "Wandb run name (default: cycle-lightning)"}
    )


def main():
    hfparser = HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments, LightningArguments
    ))

    model_args, data_args, training_args, generation_args, lightning_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    with open('configs/model_A_config.yaml', 'r') as f:
        model_A_config = yaml.safe_load(f)

    with open('configs/model_B_config.yaml', 'r') as f:
        model_B_config = yaml.safe_load(f)

    config = {
        'model_A': model_A_config,
        'model_B': model_B_config,
    }

    with open('configs/lightning_config.yaml', 'r') as f:
        lightning_config = yaml.safe_load(f)

    seed_everything(lightning_config.seed)

    model = CycleModel(config)

    trainer = Trainer(
        default_root_dir=os.path.join(lightning_config.output_dir + '/checkpoints'),
        max_epochs=lightning_config.num_epochs,
        log_every_n_steps=lightning_config.log_every_n_steps,
        val_check_interval=lightning_config.val_check_interval,
    )
    if lightning_config.use_wandb:
        trainer.logger = WandbLogger(project=lightning_config.wandb_project, entity=lightning_config.wandb_entity, name=lightning_config.wandb_run_name)
    trainer.fit(model)

    # Save the models separately for use in inference
    model.save_pretrained(lightning_config.output_dir)

if __name__ == "__main__":
    main()
