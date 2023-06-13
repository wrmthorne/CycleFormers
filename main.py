import argparse
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCasualLM

class CycleModel(LightningModule):
    def __init__(self, model_a_name_or_path, task_a, data_a, data_b, model_b_name_or_path=None, task_b=None, batch_size=32):
        super().__init__()
        
        if not task_b:
            task_b = task_a

        if not model_b_name_or_path:
            model_b_name_or_path = model_a_name_or_path
        
        if task_a.upper() == 'CAUSAL_LM':
            self.model_a = AutoModelForCasualLM.from_pretrained(model_a_name_or_path)
        elif task_a.upper() == 'SEQ2SEQ_LM':
            self.model_a = AutoModelForSeq2SeqLM.from_pretrained(model_a_name_or_path)

        if task_b.upper() == 'CAUSAL_LM':
            self.model_b = AutoModelForCasualLM.from_pretrained(model_b_name_or_path)
        elif task_b.upper() == 'SEQ2SEQ_LM':
            self.model_b = AutoModelForSeq2SeqLM.from_pretrained(model_b_name_or_path)

        self.tokenizer_a = AutoTokenizer.from_pretrained(model_a_name_or_path)
        self.tokenizer_b = AutoTokenizer.from_pretrained(model_b_name_or_path)
        self.data_a = data_a
        self.data_b = data_b
        self.batch_size = batch_size

    def configure_optimizers(self):
        opt_a = Adam(
            self.model_a.paramerters(),
            lr=2e-4, betas=(0.5, 0.999))
        
        opt_b = Adam(
            self.model_a.paramerters(),
            lr=2e-4, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        sch_a = LambdaLR(opt_a, lr_lambda=gamma)
        sch_b = LambdaLR(opt_b, lr_lambda=gamma)
        return [opt_a, opt_b], [sch_a, sch_b]
    
    def _cycle(self, batch, generating_model, training_model, training_tokenizer):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        generated_ids = generating_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        
        generated_ids = generated_ids[:, 1:]
        attention_mask = generated_ids.clone()
        attention_mask[attention_mask != 0] = 1

        labels[labels==training_tokenizer.pad_token_id] = -100

        outputs = training_model(
            input_ids      = generated_ids.to(self.device),
            attention_mask = attention_mask.to(self.device),
            labels         = labels.to(self.device)
        )

        return outputs
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss_a = self._cycle(batch, self.model_b, self.model_a, self.tokenizer_a).loss
            return {'loss': loss_a}
        elif optimizer_idx == 1:
            loss_b = self._cycle(batch, self.model_a, self.model_b, self.tokenizer_b).loss
            return {'loss ': loss_b}

    def train_dataloader(self):
        loader_a = DataLoader(load_from_disk(self.data_a), self.batch_size, shuffle=True)
        loader_b = DataLoader(load_from_disk(self.data_b), self.batch_size, shuffle=True)
        return {"a": loader_a, "b": loader_b}

def main(args):
    seed_everything(args.seed)

    model = CycleModel(
        args.model_a_name_or_path,
        args.task_a,
        args.data_a,
        args.data_b,
        args.model_b_name_or_path,
        args.task_b,
        args.batch_size
    )

    trainer = Trainer.from_argparse_args(args)
    trainer.logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a_name_or_path', type=str, default='gpt2')
    parser.add_argument('--model_b_name_or_path', type=str, default='gpt2')
    parser.add_argument('--data_a', type=str, default='data/a')
    parser.add_argument('--data_b', type=str, default='data/b')
    parser.add_argument('--task_a', type=str, default='causal_lm')
    parser.add_argument('--task_b', type=str, default='causal_lm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='CycleTraining')
    parser.add_argument('--wandb_entity', type=str, default='wandb')
    parser.add_argument('--wandb_run_name', type=str, default='CycleTraining')
    args = parser.parse_args()
    main(args)