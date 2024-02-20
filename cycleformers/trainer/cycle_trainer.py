from cycleformers import DataCollatorForCausalLM, TrainDataset
from datasets import Dataset
from .model_config import ModelConfig
from .trainer_config import TrainerConfig
from lightning.pytorch.utilities import CombinedLoader
from pytorch_lightning import LightningModule, Trainer
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, DataCollator
from typing import Optional

from ..core import TrainCycle

# TODO find more permanent solution to this
def task_type(model):
    if getattr(model.config, 'is_encoder_decoder', False):
        return 'SEQ2SEQ_LM'
    else:
        return 'CAUSAL_LM'
    

class CycleTrainer(Trainer):
    def __init__(
        self,
        model,
        trainer_config: TrainerConfig = None,
        data_collator_A: Optional[DataCollator] = None,
        data_collator_B: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        if trainer_config is None:
            trainer_config = TrainerConfig()

        super().__init__(trainer_config.to_dict())

        self.model = model

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

    def train(self):
        self.fit(self.model, train_dataloaders=self.train_dataloaders, val_dataloaders=self.val_dataloaders)

    