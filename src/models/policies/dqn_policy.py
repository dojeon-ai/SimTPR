import torch.nn as nn
import torch
from .base import BasePolicy


class DQNPolicy(BasePolicy):
    name = 'dqn'
    def __init__(self, 
                 in_features,
                 hid_features,
                 action_size):
        super().__init__()
        self.fc_q = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=action_size)
        )

    def forward(self, x):
        q = self.fc_q(x)
        return q