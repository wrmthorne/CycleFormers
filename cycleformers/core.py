from enum import Enum
import numpy as np
import random
import torch

class TrainCycle(Enum):
    '''
    Enum for the training cycle
    '''
    def __repr__(self) -> str:
        return self.value
    
    A = 'A'
    B = 'B'


def seed_everything(seed: int):
    '''
    Seed everything for reproducibility
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)