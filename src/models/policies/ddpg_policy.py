import math
import numpy as np
import torch
import torch.nn as nn


from src.common.train_utils import to_np, dmc_weight_init, gaussian_logprob, squash, SquashedNormal
import src.common.augmentation as rad
from src.models.policies.base import BasePolicy 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.train_utils import weight_init, TruncatedNormal



class DDPGPolicy(BasePolicy):
    name = 'ddpg_policy'
    def __init__(self, repr_features, feat_features, hid_features, action_shape):
        super().__init__()
        self.action_shape = action_shape
        self.hid_features = hid_features

        self.actor = Actor(repr_features, feat_features, hid_features, action_shape)
        self.critic = Critic(repr_features, feat_features, hid_features, action_shape)
        self.critic_target = Critic(repr_features, feat_features, hid_features, action_shape)
        
    def forward(self, encoded_obs, std):
        dist = self.actor(encoded_obs, std)
        return dist

    
        

class Actor(nn.Module):
    def __init__(self, repr_features, feat_features, hid_features, action_shape):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_features, feat_features),
                                   nn.LayerNorm(feat_features), nn.Tanh())

        self.policy = nn.Sequential(
            nn.Linear(feat_features, hid_features), nn.ReLU(inplace=True),
            nn.Linear(hid_features, hid_features), nn.ReLU(inplace=True),
            nn.Linear(hid_features, action_shape[0])
        )

        self.apply(weight_init)


    def forward(self, encoded_obs, std):
        h = self.trunk(encoded_obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist



class Critic(nn.Module):
    def __init__(self, repr_features, feat_features, hid_features, action_shape):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_features, feat_features),
                                   nn.LayerNorm(feat_features), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feat_features + action_shape[0], hid_features),
            nn.ReLU(inplace=True), nn.Linear(hid_features, hid_features),
            nn.ReLU(inplace=True), nn.Linear(hid_features, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feat_features + action_shape[0], hid_features),
            nn.ReLU(inplace=True), nn.Linear(hid_features, hid_features),
            nn.ReLU(inplace=True), nn.Linear(hid_features, 1))

        self.apply(weight_init)

    def forward(self, encoded_obs, action):
        h = self.trunk(encoded_obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


