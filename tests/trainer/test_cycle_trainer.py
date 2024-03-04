import gc
import unittest

import torch
from transformers import AutoTokenizer

from cycleformers.core import seed_everything
from cycleformers.trainer import CycleTrainer, ModelTrainingArguments, TrainingArguments
from ..testing_utils import (
    BaseModel,
    BaseModelWithHead,
    require_peft,
)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, query_data, response_data):
        self.query_data = query_data
        self.response_data = response_data

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        return self.query_data[idx], self.response_data[idx]



class CycleTrainerTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        seed_everything(42)

        cls.gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')

    def tearDown(self):
        gc.collect()

    def _init_dummy_dataset(self):
        query_txt_A = self.gpt2_tokenizer('Query A', return_tensors='pt')
        response_txt_A = self.gpt2_tokenizer('Response A', return_tensors='pt')
        query_txt_B = self.gpt2_tokenizer('Query B', return_tensors='pt')
        response_txt_B = self.gpt2_tokenizer('Response B', return_tensors='pt')

        dummmy_dataset_A = DummyDataset(query_txt_A, response_txt_A)
        dummmy_dataset_B = DummyDataset(query_txt_B, response_txt_B)

        return dummmy_dataset_A, dummmy_dataset_B
    
    def test_init_dict_models(self):
        models = {
            'A': BaseModelWithHead(BaseModelWithHead.config_class()),
            'B': BaseModelWithHead(BaseModelWithHead.config_class())
        }
    
