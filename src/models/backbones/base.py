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

    @property
    def output_dim(self):
        return self._output_dim

    def _renormalize(self, tensor, first_dim=1):
        # [params] first_dim: starting dimension to normalize the embedding
        flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
        max = torch.max(flat_tensor, first_dim, keepdim=True).values
        min = torch.min(flat_tensor, first_dim, keepdim=True).values
        flat_tensor = (flat_tensor - min)/(max - min + EPS)

        return flat_tensor.view(*tensor.shape)
