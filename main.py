import argparse
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
import os
import torch
from data_loader import TextDataset
from torch.nn.utils.rnn import pad_sequence

class CycleModel(LightningModule):
    def __init__(self, model_a_name_or_path, task_a, data_a, data_b, model_b_name_or_path=None, task_b=None, batch_size=32):
        super().__init__()
        
        if not task_b:
            task_b = task_a

        print((task_a, task_b))

        if not model_b_name_or_path:
            model_b_name_or_path = model_a_name_or_path
        
        if task_a.upper() == 'CAUSAL_LM':
            self.model_a = AutoModelForCausalLM.from_pretrained(model_a_name_or_path)
        elif task_a.upper() == 'SEQ2SEQ_LM':
            self.model_a = AutoModelForSeq2SeqLM.from_pretrained(model_a_name_or_path)
        else:
            raise ValueError(f'Unknown task {task_a}')

        if task_b.upper() == 'CAUSAL_LM':
            self.model_b = AutoModelForCausalLM.from_pretrained(model_b_name_or_path)
        elif task_b.upper() == 'SEQ2SEQ_LM':
            print('t5')
            self.model_b = AutoModelForSeq2SeqLM.from_pretrained(model_b_name_or_path)
        else:
            raise ValueError(f'Unknown task {task_b}')

        self.tokenizer_a = AutoTokenizer.from_pretrained(model_a_name_or_path)
        self.tokenizer_b = AutoTokenizer.from_pretrained(model_b_name_or_path)

        if not self.tokenizer_a.pad_token:
            self.tokenizer_a.pad_token = self.tokenizer_a.eos_token
        if not self.tokenizer_b.pad_token:
            self.tokenizer_b.pad_token = self.tokenizer_b.eos_token

        self.data_a = data_a
        self.data_b = data_b
        self.batch_size = batch_size

    def configure_optimizers(self):
        opt_a = Adam(
            self.model_a.parameters(),
            lr=2e-4, betas=(0.5, 0.999))
        
        opt_b = Adam(
            self.model_b.parameters(),
            lr=2e-4, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        sch_a = LambdaLR(opt_a, lr_lambda=gamma)
        sch_b = LambdaLR(opt_b, lr_lambda=gamma)
        return [opt_a, opt_b], [sch_a, sch_b]
    
    def _cycle(self, batch, generating_model, training_model, generating_tokenizer, training_tokenizer):
        print(batch)
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        generated_ids = generating_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        
        # Decode and re-encode tokens to training tokenizer
        decoded_ids = generating_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_input_ids = generating_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        training_ids = training_tokenizer.batch_encode_plus(decoded_ids, padding=True, return_tensors='pt')['input_ids']
        labels = training_tokenizer.batch_encode_plus(decoded_input_ids, padding=True, return_tensors='pt')['input_ids']

        # Set pad tokens to be ignored by PyTorch loss function
        labels[labels==training_tokenizer.pad_token_id] = -100

        # Pad sequences to same length
        padded_sequences = []
        for sequence in [training_ids, labels]:
            padded_sequences.append(pad_sequence(sequence,
                                         padding_value=training_tokenizer.pad_token_id))

        training_ids, labels = pad_sequence(padded_sequences,
                                        padding_value=training_tokenizer.pad_token_id).permute(1, 0, 2)
        
        # Create attention mask
        attention_mask = training_ids.clone()
        attention_mask[attention_mask != 0] = 1

        outputs = training_model(
            input_ids      = training_ids.to(self.device),
            attention_mask = attention_mask.to(self.device),
            labels         = labels.to(self.device)
        )

        return outputs
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss_a = self._cycle(batch['a'], self.model_b, self.model_a, self.tokenizer_b, self.tokenizer_a).loss
            return loss_a
        elif optimizer_idx == 1:
            loss_b = self._cycle(batch['b'], self.model_a, self.model_b, self.tokenizer_a, self.tokenizer_b).loss
            return loss_b
        else:
            raise ValueError(f'Unknown optimizer index {optimizer_idx}')

    def train_dataloader(self):
        self.collator_a = DataCollatorForSeq2Seq(self.tokenizer_a, return_tensors='pt', padding=True)
        self.collator_b = DataCollatorForSeq2Seq(self.tokenizer_b, return_tensors='pt', padding=True)

        dataset_a = TextDataset(self.tokenizer_a, self.tokenizer_b).load_data(self.data_a)
        dataset_b = TextDataset(self.tokenizer_b, self.tokenizer_a).load_data(self.data_b)

        loader_a = DataLoader(dataset_a, self.batch_size, shuffle=True, collate_fn=self.collator_a)
        loader_b = DataLoader(dataset_b, self.batch_size, shuffle=True, collate_fn=self.collator_b)
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
    if args.use_wandb:
        trainer.logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a_name_or_path', type=str, default='gpt2')
    parser.add_argument('--model_b_name_or_path', type=str, default=None)
    parser.add_argument('--data_a', type=str, default='data/a.json')
    parser.add_argument('--data_b', type=str, default='data/b.json')
    parser.add_argument('--task_a', type=str, default='causal_lm')
    parser.add_argument('--task_b', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='CycleTraining')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default='CycleTraining')
    args = parser.parse_args()
    main(args)