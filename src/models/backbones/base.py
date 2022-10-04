from abc import *
import torch.nn as nn
import torch
EPS = 1e-5

class BaseBackbone(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    @abstractmethod
    def forward(self, x):
        """
        [param] x (torch.Tensor): (n, t, c, h, w)
        [return] x (torch.Tensor): (n, t, d)
        """
        pass
    
    @property
    def output_dim(self):
        pass