from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from utils.datasets import TrainDataset
from utils.load_model import load_model
from lightning.pytorch.utilities import CombinedLoader
from cycles import initialise_cycle


class CycleModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.model_A, self.tokenizer_A, self.collator_A = load_model(self.hparams.model_A)
        self.model_B, self.tokenizer_B, self.collator_B = load_model(self.hparams.model_B)

        self.generation_config_A = GenerationConfig(**self.hparams.model_A['generation'])
        self.generation_config_B = GenerationConfig(**self.hparams.model_B['generation'])

        self.train_data_A, self.val_data_A = TrainDataset(self.tokenizer_A,task=self.hparams.model_A['task']).load_data(self.hparams.model_A['data']['path'])
        self.train_data_B, self.val_data_B = TrainDataset(self.tokenizer_B, task=self.hparams.model_B['task']).load_data(self.hparams.model_B['data']['path'])

        self.cycle_A = initialise_cycle(
            self.model_A,
            self.tokenizer_A,
            self.generation_config_A,
            self.hparams.model_A['task'])
        
        self.cycle_B = initialise_cycle(
            self.model_B,
            self.tokenizer_B,
            self.generation_config_B,
            self.hparams.model_B['task'])

        self.automatic_optimization = False

        if self.tokenizer_A.get_vocab() == self.tokenizer_B.get_vocab() and type(self.model_A) == type(self.model_B):
            print('Same model and tokenizer detected. Using fast cycle. This can be manually disabled in the lightning config.')
        
    def configure_optimizers(self):
        opt_A = Adam(
            self.model_A.parameters(),
            lr=float(self.hparams.model_A['learning_rate']),
            betas=(self.hparams.model_A['adam_beta1'], self.hparams.model_A['adam_beta2']))
        
        opt_B = Adam(
            self.model_B.parameters(),
            lr=float(self.hparams.model_B['learning_rate']),
            betas=(self.hparams.model_B['adam_beta1'], self.hparams.model_B['adam_beta2']))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        sch_A = LambdaLR(opt_A, lr_lambda=gamma)
        sch_B = LambdaLR(opt_B, lr_lambda=gamma)
        return [opt_A, opt_B], [sch_A, sch_B]
    
    def cycle(self, gen_cycle, train_cycle, batch):
        if self.tokenizer_A.get_vocab() == self.tokenizer_B.get_vocab() and type(self.model_A) == type(self.model_B):
            generated = gen_cycle.generate(batch)
            formated = train_cycle.format(generated)
            return train_cycle.train(formated)
        else:
            generated = gen_cycle.generate(batch)
            decoded = gen_cycle.decode(generated)
            formated = train_cycle.encode_and_format(decoded)
            return train_cycle.train(formated)

    def training_step(self, batch, batch_idx):
        opt_A, opt_B = self.optimizers()
        sch_A, sch_B = self.lr_schedulers()
        
        if batch['a']:
            loss_A = self.cycle(self.cycle_A, self.cycle_B, batch['a']).loss
            self.manual_backward(loss_A)

            if batch_idx % self.hparams.model_A['gradient_accumulation_steps'] == 0:
                opt_A.step()
                sch_A.step()
                opt_A.zero_grad()

                self.log('train_loss_A', loss_A, on_step=True, logger=True, batch_size=self.hparams.model_A['train_batch_size'])
                self.log('opt_A_lr', opt_A.param_groups[0]['lr'], on_step=True, logger=True)

        if batch['b']:
            loss_B = self.cycle(self.cycle_B, self.cycle_A, batch['b']).loss
            self.manual_backward(loss_B)

            if batch_idx % self.hparams.model_B['gradient_accumulation_steps'] == 0:
                opt_B.step()
                sch_B.step()
                opt_B.zero_grad()

                self.log('train_loss_B', loss_B, on_step=True, logger=True, batch_size=self.hparams.model_B['train_batch_size'])
                self.log('opt_B_lr', opt_B.param_groups[0]['lr'], on_step=True, logger=True)

    def validation_step(self, batch, batch_idx):
        if batch['a']:
            loss_A = self.model_A(
                input_ids=batch['a']['input_ids'],
                attention_mask=batch['a']['attention_mask'],
                labels=batch['a']['labels']
            ).loss
            self.log('val_loss_A', loss_A, on_epoch=True, logger=True, batch_size=self.hparams.model_A['eval_batch_size'])
        if batch['b']:
            loss_B = self.model_B(
                input_ids=batch['b']['input_ids'],
                attention_mask=batch['b']['attention_mask'],
                labels=batch['b']['labels']
            ).loss
            self.log('val_loss_B', loss_B, on_epoch=True, logger=True, batch_size=self.hparams.model_B['eval_batch_size'])

    def train_dataloader(self):
        loader_A = DataLoader(self.train_data_A, self.hparams.model_A['train_batch_size'], shuffle=True, collate_fn=self.collator_A)
        loader_B = DataLoader(self.train_data_B, self.hparams.model_B['train_batch_size'], shuffle=True, collate_fn=self.collator_B)
        return CombinedLoader({"a": loader_A, "b": loader_B}, 'max_size')

    def val_dataloader(self):    
        if self.val_data_A:
            loader_A = DataLoader(self.val_data_A, self.hparams.model_A['eval_batch_size'], shuffle=True, collate_fn=self.collator_A)
        else:
            loader_A = {}
        if self.val_data_B:
            loader_B = DataLoader(self.val_data_B, self.hparams.model_B['eval_batch_size'], shuffle=True, collate_fn=self.collator_B)
        else:
            loader_B = {}
        return CombinedLoader({"a": loader_A, "b": loader_B}, 'max_size')
    
    def save_pretrained(self, path):
        self.model_A.save_pretrained(path + '/model_A')
        self.model_B.save_pretrained(path + '/model_B')
        self.tokenizer_A.save_pretrained(path + '/model_A')
        self.tokenizer_B.save_pretrained(path + '/model_B')
