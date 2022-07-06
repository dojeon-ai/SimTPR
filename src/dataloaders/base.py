from abc import *
from collections import namedtuple
from torch.utils.data import DataLoader


class BaseLoader(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    def get_dataloader(self)-> DataLoader:
        pass