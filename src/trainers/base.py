import torch
import torch.nn as nn
from abc import *


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    @abstractmethod
    def _build_optimizer(self):
        pass

    def create_state_dict(self):
        return {
            'model_state_dict': self.model.module.state_dict(),
        }

    def save_state_dict(self, path):
        state_dict = {'model_state_dict': self.model.state_dict()}
        torch.save(state_dict, path)
    
    def load_state_dict(self, path):
        return torch.load(path)
    
    def train(self):
        pass

    def evaluate(self):
        pass


