import unittest

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from cycleformers.import_utils import is_peft_available


def require_peft(test_case):
    '''Decorator to skip tests if PEFT is not installed.'''
    if not is_peft_available():
        return unittest.skip('PEFT is not installed')(test_case)
    

# https://github.com/huggingface/transformers/blob/main/tests/test_modeling_utils.py
class BaseModel(PreTrainedModel):
    base_model_prefix = "base"
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(5, 5)
        self.linear_2 = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear_2(self.linear(x))
    
# https://github.com/huggingface/transformers/blob/main/tests/test_modeling_utils.py
class BaseModelWithHead(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def _init_weights(self, module):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.base = BaseModel(config)
            # linear is a common name between Base and Head on purpose.
            self.linear = nn.Linear(5, 5)
            self.linear2 = nn.Linear(5, 5)

        def forward(self, x):
            return self.linear2(self.linear(self.base(x)))