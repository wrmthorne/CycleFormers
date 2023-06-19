import argparse
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
import torch
from data_loader import TextDataset
from torch.nn.utils.rnn import pad_sequence
import sys
from lightning.pytorch.utilities import CombinedLoader

class CycleModel(LightningModule):
    def __init__(self, model_a_name_or_path, task_a, data_a, data_a_val_size, data_b, data_b_val_size, model_b_name_or_path=None, task_b=None, batch_size=32):
        super().__init__()
        
        self.task_a = task_a
        if not task_b:
            self.task_b = task_a

        if not model_b_name_or_path:
            model_b_name_or_path = model_a_name_or_path
        
        if self.task_a.upper() == 'CAUSAL_LM':
            self.model_a = AutoModelForCausalLM.from_pretrained(model_a_name_or_path)
        elif self.task_a.upper() == 'SEQ2SEQ_LM':
            self.model_a = AutoModelForSeq2SeqLM.from_pretrained(model_a_name_or_path)
        else:
            raise ValueError(f'Unknown task {self.task_a}')

        if self.task_b.upper() == 'CAUSAL_LM':
            self.model_b = AutoModelForCausalLM.from_pretrained(model_b_name_or_path)
        elif self.task_b.upper() == 'SEQ2SEQ_LM':
            print('t5')
            self.model_b = AutoModelForSeq2SeqLM.from_pretrained(model_b_name_or_path)
        else:
            raise ValueError(f'Unknown task {self.task_b}')

        self.tokenizer_a = AutoTokenizer.from_pretrained(model_a_name_or_path)
        self.tokenizer_b = AutoTokenizer.from_pretrained(model_b_name_or_path)

        if not self.tokenizer_a.pad_token:
            self.tokenizer_a.pad_token = self.tokenizer_a.eos_token
        if not self.tokenizer_b.pad_token:
            self.tokenizer_b.pad_token = self.tokenizer_b.eos_token

        self.collator_a = DataCollatorForSeq2Seq(self.tokenizer_a, return_tensors='pt', padding=True)
        self.collator_b = DataCollatorForSeq2Seq(self.tokenizer_b, return_tensors='pt', padding=True)

        self.train_data_a, self.val_data_a = TextDataset(self.tokenizer_a).load_data(data_a, data_a_val_size)
        self.train_data_b, self.val_data_b = TextDataset(self.tokenizer_b).load_data(data_b, data_b_val_size)

        print(self.val_data_a)
        print(self.val_data_b)
        self.batch_size = batch_size

        self.automatic_optimization = False

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
    

    def _cycle_causal(self, batch, generating_model, training_model, generating_tokenizer, training_tokenizer):
        # Input --GenModel.generate>> Input + ResponseA
        # ResponseA + Input --TrainModel.__call__>> Reconstruction Loss
        prompt_gen, attention_mask = batch['input_ids'], batch['attention_mask']
        prompt_response_gen = generating_model.generate(
            input_ids      = prompt_gen.to(self.device),
            attention_mask = attention_mask.to(self.device),
            )
        
        # Split response from prompt
        response_gen = prompt_response_gen[:, prompt_gen.shape[-1]:]

        # Switch generator models prompt and response to be new labels
        labels = torch.cat((response_gen.to(self.device), prompt_gen), dim=1)
        labels[labels==self.tokenizer.pad_token_id] = -100

        # pad new input to match label size
        pad_size = labels.shape[-1] - response_gen.shape[-1]
        pad_tensor = torch.full((labels.shape[0], pad_size), training_tokenizer.pad_token_id)
        prompt_train = torch.cat((pad_tensor.to(self.device), response_gen), dim=-1)
        attention_mask_train = torch.ones_like(prompt_train)

        outputs = training_model(
            input_ids      = prompt_train.to(self.device),
            attention_mask = attention_mask_train.to(self.device),
            labels         = labels.to(self.device)
            )
        
        return outputs
    
    def _cycle_seq2seq(self, batch, generating_model, training_model, generating_tokenizer, training_tokenizer):
        input_gen, attention_mask_gen = batch['input_ids'], batch['attention_mask']
        response_gen = generating_model.generate(
            input_ids      = input_gen.to(self.device),
            attention_mask = attention_mask_gen.to(self.device),
            )
        
        input_train = response_gen[:, 1:] # remove leading pad token
        attention_mask_train = torch.ones_like(input_train)
        labels = input_gen
        labels[labels==training_tokenizer.pad_token_id] = -100

        outputs = training_model(
            input_ids      = input_train.to(self.device),
            attention_mask = attention_mask_train.to(self.device),
            labels         = labels.to(self.device)
            )
        
        return outputs

    def training_step(self, batch, batch_idx):
        opt_a, opt_b = self.optimizers()
        sch_a, sch_b = self.lr_schedulers()
        
        if batch['a']:
            loss_a = self._cycle_seq2seq(batch['a'], self.model_b, self.model_a, self.tokenizer_b, self.tokenizer_a).loss
            self.manual_backward(loss_a)
            opt_a.step()
            sch_a.step()
            opt_a.zero_grad()

            self.log('train_loss_a', loss_a, on_step=True, logger=True)
            self.log('opt_a_lr', opt_a.param_groups[0]['lr'], on_step=True, logger=True)

        if batch['b']:
            loss_b = self._cycle_seq2seq(batch['b'], self.model_a, self.model_b, self.tokenizer_a, self.tokenizer_b).loss
            self.manual_backward(loss_b)
            opt_b.step()
            sch_b.step()
            opt_b.zero_grad()

            self.log('train_loss_b', loss_b, on_step=True, logger=True)
            self.log('opt_b_lr', opt_b.param_groups[0]['lr'], on_step=True, logger=True)

    def train_dataloader(self):
        loader_a = DataLoader(self.train_data_a, self.batch_size, shuffle=True, collate_fn=self.collator_a)
        loader_b = DataLoader(self.train_data_b, self.batch_size, shuffle=True, collate_fn=self.collator_b)
        return CombinedLoader({"a": loader_a, "b": loader_b}, 'max_size')

    def val_dataloader(self):
        loader_a = DataLoader(self.val_data_a, self.batch_size, shuffle=True, collate_fn=self.collator_a)
        loader_b = DataLoader(self.val_data_b, self.batch_size, shuffle=True, collate_fn=self.collator_b)
        return CombinedLoader({"a": loader_a, "b": loader_b}, 'max_size')

def main(args):
    seed_everything(args.seed)

    model = CycleModel(
        args.model_a_name_or_path,
        args.task_a,
        args.data_a,
        args.data_a_val_size,
        args.data_b,
        args.data_b_val_size,
        args.model_b_name_or_path,
        args.task_b,
        args.batch_size
    )

    # if torch.__version__ >= "2.0.0" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer = Trainer(
        max_epochs=10,
        log_every_n_steps=1
    )
    if args.use_wandb:
        trainer.logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a_name_or_path', type=str, default='gpt2', help='Model name or path to model for model A (default: gpt2)')
    parser.add_argument('--model_b_name_or_path', type=str, default=None, help='Model name or path to model for model B (default: gpt2)')
    parser.add_argument('--data_a', type=str, default='data/a.json', help='Path to data for model A (default: data/a.json))')
    parser.add_argument('--data_b', type=str, default='data/b.json', help='Path to data for model B (default: data/b.json)))')
    parser.add_argument('--data_a_val_size', type=float, default=0, help='Validation size for data A if no validation split is provided (default: 0)')
    parser.add_argument('--data_b_val_size', type=float, default=0, help='Validation size for data B if no validation split is provided (default: 0)')
    parser.add_argument('--task_a', type=str, default='causal_lm', help='Task for model A (default: causal_lm))')
    parser.add_argument('--task_b', type=str, default=None, help='Task for model B (default: causal_lm)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use wandb for logging (default: False)')
    parser.add_argument('--wandb_project', type=str, default='CycleTraining', help='Wandb project name (default: CycleTraining)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name (default: None)')
    parser.add_argument('--wandb_run_name', type=str, default='CycleTraining', help='Wandb run name (default: CycleTraining)')
    args = parser.parse_args()
    main(args)