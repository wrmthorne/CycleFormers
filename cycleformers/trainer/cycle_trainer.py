from cycleformers import DataCollatorForCausalLM, TrainDataset
from lightning.pytorch.utilities import CombinedLoader
from pytorch_lightning import LightningModule, Trainer
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from ..core import TrainCycle

# TODO find more permanent solution to this
def task_type(model):
    if getattr(model.config, 'is_encoder_decoder', False):
        return 'SEQ2SEQ_LM'
    else:
        return 'CAUSAL_LM'

class CycleModule(LightningModule):
    def __init__(
        self,
        model,
        dataset_A,
        dataset_B,
        **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False

        self.model = model
        self.train_dataset_A, self.val_dataset_A = TrainDataset(self.model.tokenizer_a, task=task_type(self.model.model_a)).load_data(dataset_A)
        self.train_dataset_B, self.val_dataset_B = TrainDataset(self.model.tokenizer_b, task=task_type(self.model.model_a)).load_data(dataset_B)


        collator_a_class = DataCollatorForSeq2Seq if getattr(self.model.model_a.config, 'is_encoder_decoder', False) else DataCollatorForCausalLM
        collator_b_class = DataCollatorForSeq2Seq if getattr(self.model.model_b.config, 'is_encoder_decoder', False) else DataCollatorForCausalLM
        self.collator_A = collator_a_class(
            tokenizer=self.model.tokenizer_a,
            return_tensors='pt',
            pad_to_multiple_of=8
        )
        self.collator_B = collator_b_class(
            tokenizer=self.model.tokenizer_b,
            return_tensors='pt',
            pad_to_multiple_of=8
        )


    def configure_optimizers(self):
        opt_A = Adam(
            self.model.model_a.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999)
        )
        
        opt_B = Adam(
            self.model.model_b.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999)
        )
        
        # TODO: Add support for custom learning rate schedulers
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        sch_A = LambdaLR(opt_A, lr_lambda=gamma)
        sch_B = LambdaLR(opt_B, lr_lambda=gamma)
        return [opt_A, opt_B], [sch_A, sch_B]
        
    def training_step(self, batch, batch_idx):
        opt_A, opt_B = self.optimizers()
        sch_A, sch_B = self.lr_schedulers()
        
        batch, _, _ = batch

        if batch[TrainCycle.A]:
            loss_A = self.model.cycle_a(batch[TrainCycle.A]).loss
            self.manual_backward(loss_A)

            opt_A.step()
            sch_A.step()
            opt_A.zero_grad()

        if batch[TrainCycle.B]:
            loss_B = self.model.cycle_b(batch[TrainCycle.B]).loss
            self.manual_backward(loss_B)

            opt_B.step()
            sch_B.step()
            opt_B.zero_grad()

    def validation_step(self, batch, batch_idx):
        batch, _, _ = batch

        if batch[TrainCycle.A]:
            loss_A = self.model.model_a(
                input_ids=batch[TrainCycle.A]['input_ids'].to(self.model.model_a.device),
                attention_mask=batch[TrainCycle.A]['attention_mask'].to(self.model.model_a.device),
                labels=batch[TrainCycle.A]['labels'].to(self.model.model_a.device),
            ).loss
        if batch[TrainCycle.B]:
            loss_B = self.model.model_b(
                input_ids=batch[TrainCycle.B]['input_ids'].to(self.model.model_b.device),
                attention_mask=batch[TrainCycle.B]['attention_mask'].to(self.model.model_b.device),
                labels=batch[TrainCycle.B]['labels'].to(self.model.model_b.device),
            ).loss

    def train_dataloader(self):
        loader_A = DataLoader(self.train_dataset_A, 4, shuffle=True, collate_fn=self.collator_A)
        loader_B = DataLoader(self.train_dataset_B, 4, shuffle=True, collate_fn=self.collator_B)
        return CombinedLoader({TrainCycle.A: loader_A, TrainCycle.B: loader_B}, 'max_size')

    def val_dataloader(self):    
        if self.val_dataset_A:
            loader_A = DataLoader(self.val_dataset_A, 4, shuffle=True, collate_fn=self.collator_A)
        else:
            loader_A = {}
        if self.val_dataset_B:
            loader_B = DataLoader(self.val_dataset_B, 4, shuffle=True, collate_fn=self.collator_B)
        else:
            loader_B = {}
        return CombinedLoader({TrainCycle.A: loader_A, TrainCycle.B: loader_B}, 'max_size')

    