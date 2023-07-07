import os
from datasets import load_from_disk, load_dataset

class TrainDataset(object):
    def __init__(self, tokenizer, task='CAUSAL_LM', max_length=512):
        super(TrainDataset, self).__init__()

        self.max_length = max_length
        self.task = task.upper()
        self.tokenizer = tokenizer

    def tokenize(self, text):
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        return result
    
    def build_labels(self, example):
        if self.task == 'CAUSAL_LM':
            return self.tokenize(example['text'] + ' ' + example['label'])['input_ids']
        elif self.task == 'SEQ2SEQ_LM':
            return self.tokenize(example['label'])['input_ids']
        else:
            raise ValueError(f'Unknown task {self.task}. Must be one of "causal_lm" or "seq2seq_lm"')

    def load_data(self, data_path):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            data = load_dataset('json', data_files=data_path)
        elif os.path.isdir(data_path):
            data = load_from_disk(data_path)
        else:
            load_dataset(data_path)

        data['train'] = data['train'].map(lambda x: self.tokenize(x['text']), remove_columns=['text'])
        data['train'] = data['train'].filter(lambda x: len(x['input_ids']) > 0)

        if 'validation' in data.keys():
            data['validation'] = data['validation'].map(lambda x: {
                    **self.tokenize(x['text']),
                    'labels': self.build_labels(x)
                }, remove_columns=['text', 'label'])
            data['validation'] = data['validation'].filter(lambda x: len(x['input_ids']) > 0)
        else:
            data['validation'] = None

        return data['train'], data['validation']
    

class InferenceDataset(object):
    def __init__(self, tokenizer, max_length=512):
        super(InferenceDataset, self).__init__()

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