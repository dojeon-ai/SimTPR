import torch
import torch.nn as nn
from abc import *


class BaseAgent(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    @abstractmethod
    def _build_optimizer(self):
        pass

    def predict(self):
        pass
    
    def train(self):
        pass

    def evaluate(self):
        pass


