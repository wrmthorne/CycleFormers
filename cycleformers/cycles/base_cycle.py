import torch

class Cycle:
    @staticmethod
    def generate(batch):
        raise NotImplementedError
    
    @staticmethod
    def decode(batch):
        raise NotImplementedError
    
    @staticmethod
    def encode(batch):
        raise NotImplementedError
    
    @staticmethod
    def format(batch):
        raise NotImplementedError
    
    @staticmethod
    def train(batch, labels):
        raise NotImplementedError