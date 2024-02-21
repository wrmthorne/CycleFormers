from enum import Enum

class TrainCycle(Enum):
    '''
    Enum for the training cycle
    '''
    def __repr__(self) -> str:
        return self.value
    
    A = 'A'
    B = 'B'