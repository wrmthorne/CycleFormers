import os
from datasets import load_from_disk, load_dataset

class TrainDataset(object):
    def __init__(self, tokenizer, max_length=512):
        super(TrainDataset, self).__init__()

        self.max_length = max_length
        self.tokenizer = tokenizer

    def tokenize(self, text):
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        return result

    def load_data(self, data_path):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            data = load_dataset('json', data_files=data_path)
        elif os.path.isdir(data_path):
            data = load_from_disk(data_path)
        else:
            load_dataset(data_path)

        data['train'] = data['train'].map(lambda x: self.tokenize(x['text']), remove_columns=['text'])

        if 'validation' in data.keys():
            data['validation'] = data['validation'].map(lambda x: {
                    **self.tokenize(x['text']),
                    'labels': self.tokenize(x['label'])['input_ids']
                }, remove_columns=['text', 'label'])
        else:
            data['validation'] = None

        return data['train'], data['validation']
    

class TestDataset(object):
    def __init__(self, tokenizer, max_length=512):
        super(TestDataset, self).__init__()

        self.max_length = max_length
        self.tokenizer = tokenizer

    def tokenize(self, text):
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        return result
    
    def load_data(self, data_path):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            data = load_dataset('json', data_files=data_path)
        elif os.path.isdir(data_path):
            data = load_from_disk(data_path)
        else:
            load_dataset(data_path)
        
        if 'test' not in data.keys():
            data['test'] = data['train']
        
        data['test'] = data['test'].map(lambda x: self.tokenize(x['text']), remove_columns=['text'])

        return data['test']