import os
from datasets import load_from_disk, load_dataset
from dataclasses import dataclass
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


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
    
@dataclass
class DataCollatorForCausalLM:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["input_ids"]))
                attn_remainder = [0] * (max_label_length - len(feature["attention_mask"]))
                if isinstance(feature["input_ids"], list):
                    feature["input_ids"] = (
                        feature["input_ids"] + remainder if padding_side == "right" else remainder + feature["input_ids"]
                    )
                    feature["attention_mask"] = (
                        feature["attention_mask"] + attn_remainder if padding_side == "right" else attn_remainder + feature["attention_mask"]
                    )
                elif padding_side == "right":
                    feature["input_ids"] = np.concatenate([feature["input_ids"], remainder]).astype(np.int64)
                    feature['attention_mask'] = np.concatenate([feature['attention_mask'], np.zeros_like(remainder)]).astype(np.int64)
                else:
                    feature["input_ids"] = np.concatenate([remainder, feature["input_ids"]]).astype(np.int64)
                    feature['attention_mask'] = np.concatenate([np.zeros_like(remainder), feature['attention_mask']]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features



