from datasets import Dataset
from lightning.pytorch.utilities import CombinedLoader
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import DataCollator
from typing import Optional

from .trainer_config import TrainerConfig
from .trainer_utils import prepare_data_collator, validate_train_dataset, validate_data_collator

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
        model = None,
        args: TrainerConfig = None,
        data_collator_A: Optional[DataCollator] = None,
        data_collator_B: Optional[DataCollator] = None,
        train_dataset_A: Optional[Dataset] = None,
        train_dataset_B: Optional[Dataset] = None,
        eval_dataset_A: Optional[Dataset] = None,
        eval_dataset_B: Optional[Dataset] = None,
    ):
        if args is None:
            args = TrainerConfig()

        self.args = args

        super().__init__(**args.to_dict())

        # if not isinstance(model, CycleModel):
        #     raise ValueError(f'model must be a CycleModel, got {type(model)}')
        self._model = model

        self.train_dataset_A = train_dataset_A
        self.train_dataset_B = train_dataset_B
        self.eval_dataset_A = eval_dataset_A
        self.eval_dataset_B = eval_dataset_B

        validate_train_dataset(train_dataset_A, 'train_dataset_A', args.max_steps)
        validate_train_dataset(train_dataset_B, 'train_dataset_B', args.max_steps)

        self.data_collator_A = prepare_data_collator(model.model_A, model.tokenizer_A, data_collator_A)
        self.data_collator_B = prepare_data_collator(model.model_B, model.tokenizer_B, data_collator_B)

        validate_data_collator(self.data_collator_A, 'data_collator_A')
        validate_data_collator(self.data_collator_B, 'data_collator_B')


    def train_dataloader(self):
        loader_A = DataLoader(
            self.train_dataset_A,
            self._model.model_A_config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator_A
        )
        loader_B = DataLoader(
            self.train_dataset_B,
            self._model.model_B_config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator_B
        )
        return CombinedLoader({TrainCycle.A: loader_A, TrainCycle.B: loader_B}, 'max_size')

    def val_dataloader(self):    
        if self.eval_dataset_A:
            loader_A = DataLoader(
                self.eval_dataset_A,
                self._model.model_A_config.per_device_eval_batch_size,
                shuffle=True,
                collate_fn=self.data_collator_A
            )
        else:
            loader_A = {}
        if self.eval_dataset_B:
            loader_B = DataLoader(
                self.eval_dataset_B,
                self._model.model_B_config.per_device_eval_batch_size,
                shuffle=True,
                collate_fn=self.data_collator_B
            )
        else:
            loader_B = {}
        return CombinedLoader({TrainCycle.A: loader_A, TrainCycle.B: loader_B}, 'max_size')

    def train(self):
        self.fit(self._model, train_dataloaders=self.train_dataloader(), val_dataloaders=self.val_dataloader())

    