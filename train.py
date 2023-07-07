import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from CycleModel import CycleModel
from transformers import HfArgumentParser, GenerationConfig
from dataclasses import dataclass, field
from typing import Optional

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
    bits: int = field(
        default=None,
        metadata={"help": "Quantise model to n bits (default: None)"}
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
    modules: str = field(
        default=None,
        metadata={"help": "Which modules to add LoRA modules to (default: All linear)"}
    )
    lora_bias: str = field(
        default=None,
        metadata={"help": 'Whether to use bias in the lora layers (default: None)'}
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

    seed_everything(args.seed)

    print(f'Unused arguments: {extra_args}')

    model = CycleModel(
        **args.__dict__,
        generation_config=generation_config,
    )

    trainer = Trainer(
        default_root_dir=args.output_dir + '/checkpoints',
        max_epochs=lightning_args.num_epochs,
        log_every_n_steps=lightning_args.log_every_n_steps,
        val_check_interval=lightning_args.val_check_interval,
    )
    if lightning_args.use_wandb:
        trainer.logger = WandbLogger(project=lightning_args.wandb_project, entity=lightning_args.wandb_entity, name=lightning_args.wandb_run_name)
    trainer.fit(model)

    # Save the models separately for use in inference
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()