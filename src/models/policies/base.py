from abc import *
import torch.nn as nn
import torch


class BasePolicy(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name
