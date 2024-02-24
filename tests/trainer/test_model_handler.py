import gc
import os
import unittest

from transformers import AutoTokenizer

from cycleformers.core import seed_everything
from cycleformers import ModelTrainingArguments, TrainingArguments
from cycleformers.trainer.model_handler import _ModelHandler
from ..testing_utils import BaseModelWithHead


class TestModelHandler(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        seed_everything(42)

        cls.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        cls.model = BaseModelWithHead(BaseModelWithHead.config_class())

    def tearDown(self):
        if os.path.exists('./test'):
            os.removedirs('./test')

        gc.collect()

    def test_model_handler_init(self):
        model_config = ModelTrainingArguments(output_dir='./test')
        model_config.update_from_global_args(TrainingArguments(output_dir='./test'))
        _ModelHandler(
            model=self.model,
            args=model_config,
            tokenizer=self.tokenizer,
            optimizers=(None, None)
        )
        