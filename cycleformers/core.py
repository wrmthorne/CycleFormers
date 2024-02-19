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