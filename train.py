import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from CycleModel import CycleModel

def main(args):
    seed_everything(args.seed)

    model = CycleModel(
        args.model_name_or_path,
        args.data_a,
        args.data_b,
        args.task,
        args.batch_size,
        args.gradient_accumulation_steps
    )

    trainer = Trainer(
        default_root_dir=args.output_dir + '/checkpoints',
        max_epochs=1,
        log_every_n_steps=1
    )
    if args.use_wandb:
        trainer.logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
    trainer.fit(model)

    # Save the models separately for use in inference
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='Model name or path to model for model A (default: gpt2)')
    parser.add_argument('--output_dir', type=str, default='models', help='Path to save model checkpoints (default: models)')
    parser.add_argument('--data_a', type=str, default='data/example/A', help='Path to data for model A. If this a validation set can be found, it will be loaded (default: data/example/A))')
    parser.add_argument('--data_b', type=str, default='data/example/B', help='Path to data for model B. If this a validation set can be found, it will be loaded (default: data/example/B)))')
    parser.add_argument('--task', type=str, default='causal_lm', help='Task for model A (default: causal_lm))')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation (default: 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use wandb for logging (default: False)')
    parser.add_argument('--wandb_project', type=str, default='CycleTraining', help='Wandb project name (default: CycleTraining)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name (default: None)')
    parser.add_argument('--wandb_run_name', type=str, default='CycleTraining', help='Wandb run name (default: CycleTraining)')
    args = parser.parse_args()
    main(args)