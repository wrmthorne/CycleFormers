import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from CycleModel import CycleModel
import yaml
import os

def main():
    parser = argparse.ArgumentParser('Training script for transformer based cycle consistency training.')
    parser.add_argument('--config_A', type=str, default='configs/model_A_config.yaml', help='Path to the config file.')
    parser.add_argument('--config_B', type=str, default='configs/model_B_config.yaml', help='Path to the config file.')
    parser.add_argument('--lightning_config', type=str, default='configs/lightning_config.yaml', help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config_A, 'r') as f:
        model_A_config = yaml.safe_load(f)

    with open(args.config_B, 'r') as f:
        model_B_config = yaml.safe_load(f)

    with open(args.lightning_config, 'r') as f:
        lightning_config = yaml.safe_load(f)

    seed_everything(lightning_config['seed'])

    model = CycleModel(model_A=model_A_config, model_B=model_B_config, use_fast_cycle=lightning_config['use_fast_cycle'])

    trainer = Trainer(
        default_root_dir=os.path.join(lightning_config['output_dir'] + '/checkpoints'),
        max_epochs=lightning_config['num_epochs'],
        log_every_n_steps=lightning_config['log_every_n_steps'],
        val_check_interval=lightning_config['val_check_interval'],
        accelerator=lightning_config['accelerator'],
        devices=lightning_config['devices'],
    )
    if lightning_config['use_wandb']:
        trainer.logger = WandbLogger(project=lightning_config['wandb_project'], entity=lightning_config['wandb_entity'], name=lightning_config['wandb_run_name'])
    trainer.fit(model)

    # Save the models separately for use in inference
    model.save_pretrained(lightning_config['output_dir'])

if __name__ == "__main__":
    main()
