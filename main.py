import argparse
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torchsummary import summary
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from datasets import load_from_disk

class CycleTraining(LightningModule):
    def __init__(self, model_a, model_b, data_a, data_b, task, batch_size=32):
        super().__init__()
        
        self.model_a = model_a
        self.model_b = model_b
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
    
    def _cycle(self, batch, generating_model, training_model):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        generated_ids = generating_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        
        generated_ids = generated_ids[:, 1:]
        attention_mask = generated_ids.clone()
        attention_mask[attention_mask != 0] = 1

        labels[labels==self.tokenizer.pad_token_id] = -100

        outputs = training_model(
            input_ids      = generated_ids.to(self.device),
            attention_mask = attention_mask.to(self.device),
            labels         = labels.to(self.device)
        )

        return outputs
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss_a = self._cycle(batch, self.model_a, self.model_b).loss
            return {'loss': loss_a}
        elif optimizer_idx == 1:
            loss_b = self._cycle(batch, self.model_a, self.model_b).loss
            return {'loss ': loss_b}

    def train_dataloader(self):
        loader_a = DataLoader(load_from_disk(self.data_a), self.batch_size, shuffle=True)
        loader_b = DataLoader(load_from_disk(self.data_b), self.batch_size, shuffle=True)       
        return {"a": loader_a, "b": loader_b}


def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a', type=str, default='bert-base-uncased')
    parser.add_argument('--model_b', type=str, default='bert-base-uncased')
    parser.add_argument('--data_a', type=str, default='data/a')
    parser.add_argument('--data_b', type=str, default='data/b')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)