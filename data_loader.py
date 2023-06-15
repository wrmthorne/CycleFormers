from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule
import os
from datasets import load_from_disk, load_dataset

class TextDataset(object):
    def __init__(self, tokenizer, label_tokenizer, max_length=512):
        super(TextDataset, self).__init__()

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.label_tokenizer = label_tokenizer

    def tokenize(self, text):
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        result['labels'] = self.label_tokenizer.encode(text)

        return result

    def load_data(self, data_path):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            data = load_dataset('json', data_files=data_path)
        elif os.path.isdir(data_path):
            data = load_from_disk(data_path)
        else:
            load_dataset(data_path)

        data['train'] = data['train'].map(lambda x: self.tokenize(x['text']), remove_columns=['text'])
        return data['train']