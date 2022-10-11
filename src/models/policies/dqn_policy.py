import torch.nn as nn
import torch
from .base import BasePolicy


class DQNPolicy(BasePolicy):
    name = 'dqn'
    def __init__(self, 
                 in_dim,
                 hid_dim,
                 action_size):
        super().__init__()
        self.fc_q = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, action_size)
        )

    def forward(self, x):
        q = self.fc_q(x)
        info = {}
        return q, info