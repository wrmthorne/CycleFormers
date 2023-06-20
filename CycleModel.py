import argparse
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
import torch
from data_loader import TextDataset
from lightning.pytorch.utilities import CombinedLoader

class CycleModel(LightningModule):
    def __init__(self, model_name_or_path, data_a, data_b, task, batch_size, gradient_accumulation_steps):
        super().__init__()

        self.task = task.upper()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.task == 'CAUSAL_LM':
            self.model_a = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            self.model_b = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            self.tokenizer.padding_side = 'left'
            self.cycle = self._cycle_causal
        elif self.task == 'SEQ2SEQ_LM':
            self.model_a = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            self.model_b = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            self.cycle = self._cycle_seq2seq
        else:
            raise ValueError(f'Unknown task {self.task}')

        self.collator_a = DataCollatorForSeq2Seq(self.tokenizer, return_tensors='pt', padding=True)
        self.collator_b = DataCollatorForSeq2Seq(self.tokenizer, return_tensors='pt', padding=True)

        self.train_data_a, self.val_data_a = TextDataset(self.tokenizer).load_data(data_a)
        self.train_data_b, self.val_data_b = TextDataset(self.tokenizer).load_data(data_b)

        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

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
    

    def _cycle_causal(self, batch, generating_model, training_model):
        # Input --GenModel.generate>> Input + ResponseA
        # ResponseA + Input --TrainModel.__call__>> Reconstruction Loss
        prompt_gen, attention_mask = batch['input_ids'], batch['attention_mask']
        prompt_response_gen = generating_model.generate(
            input_ids      = prompt_gen.to(self.device),
            attention_mask = attention_mask.to(self.device),
            pad_token_id   = self.tokenizer.pad_token_id,
            )
        
        # Split response from prompt
        response_gen = prompt_response_gen[:, prompt_gen.shape[-1]:]

        # Switch generator models prompt and response to be new labels
        labels = torch.cat((response_gen.to(self.device), prompt_gen), dim=1)
        labels[labels==self.tokenizer.pad_token_id] = -100

        # pad new input to match label size
        pad_size = labels.shape[-1] - response_gen.shape[-1]
        pad_tensor = torch.full((labels.shape[0], pad_size), self.tokenizer.pad_token_id)
        prompt_train = torch.cat((pad_tensor.to(self.device), response_gen), dim=-1)
        attention_mask_train = torch.ones_like(prompt_train)

        outputs = training_model(
            input_ids      = prompt_train.to(self.device),
            attention_mask = attention_mask_train.to(self.device),
            labels         = labels.to(self.device)
            )
        
        return outputs
    
    def _cycle_seq2seq(self, batch, generating_model, training_model):
        input_gen, attention_mask_gen = batch['input_ids'], batch['attention_mask']
        response_gen = generating_model.generate(
            input_ids      = input_gen.to(self.device),
            attention_mask = attention_mask_gen.to(self.device),
            pad_token_id   = self.tokenizer.pad_token_id,
            )
        
        input_train = response_gen[:, 1:] # remove leading pad token
        attention_mask_train = torch.ones_like(input_train)
        labels = input_gen
        labels[labels==self.tokenizer.pad_token_id] = -100

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
            loss_a = self.cycle(batch['a'], self.model_b, self.model_a).loss
            self.manual_backward(loss_a)

            if batch_idx % self.gradient_accumulation_steps == 0:
                opt_a.step()
                sch_a.step()
                opt_a.zero_grad()

                self.log('train_loss_a', loss_a, on_step=True, logger=True)
                self.log('opt_a_lr', opt_a.param_groups[0]['lr'], on_step=True, logger=True)

        if batch['b']:
            loss_b = self.cycle(batch['b'], self.model_a, self.model_b).loss
            self.manual_backward(loss_b)

            if batch_idx % self.gradient_accumulation_steps == 0:
                opt_b.step()
                sch_b.step()
                opt_b.zero_grad()

                self.log('train_loss_b', loss_b, on_step=True, logger=True)
                self.log('opt_b_lr', opt_b.param_groups[0]['lr'], on_step=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # TODO: validate whether this is the mean loss over the whole validation set or just the final batch eval loss
        if batch['a']:
            loss_a = self.model_a(**batch['a']).loss
            self.log('val_loss_a', loss_a, on_epoch=True, logger=True)
        if batch['b']:
            loss_b = self.model_b(**batch['b']).loss
            self.log('val_loss_b', loss_b, on_epoch=True, logger=True)

    def train_dataloader(self):
        loader_a = DataLoader(self.train_data_a, self.batch_size, shuffle=True, collate_fn=self.collator_a)
        loader_b = DataLoader(self.train_data_b, self.batch_size, shuffle=True, collate_fn=self.collator_b)
        return CombinedLoader({"a": loader_a, "b": loader_b}, 'max_size')

    def val_dataloader(self):
        loader_a = DataLoader(self.val_data_a, self.batch_size, shuffle=True, collate_fn=self.collator_a)
        loader_b = DataLoader(self.val_data_b, self.batch_size, shuffle=True, collate_fn=self.collator_b)
        return CombinedLoader({"a": loader_a, "b": loader_b}, 'max_size')
    
    def save_pretrained(self, path):
        self.model_a.save_pretrained(path + '/model_A')
        self.model_b.save_pretrained(path + '/model_B')
        self.tokenizer.save_pretrained(path + '/model_A')
        self.tokenizer.save_pretrained(path + '/model_B')
