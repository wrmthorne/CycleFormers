import unittest

from transformers import TrainingArguments as HFTrainingArguments

from cycleformers.trainer.training_args import ModelTrainingArguments, TrainingArguments


class TestTrainingArguments(unittest.TestCase):
    def test_training_arguments(self):
        hf_args = HFTrainingArguments(output_dir='./test')
        my_args = TrainingArguments(output_dir='./test')

        self.assertDictEqual(hf_args.to_dict(), my_args.to_dict())


# TODO: Add tests for ModelTrainingArguments