import torch
import torch.nn as nn
import numpy as np
from collections import deque
from .per_buffer import PERBuffer
from src.common.train_utils import LinearScheduler


class VITBuffer(PERBuffer):
    name = 'vit_buffer'
    def __init__(self, size, n_step, gamma, prior_exp, prior_weight_scheduler, device):
        super().__init__(size=size,
                         n_step=n_step,
                         gamma=gamma,
                         prior_exp=prior_exp,
                         prior_weight_scheduler=prior_weight_scheduler,
                         device=device)

    def encode_obs(self, obs, prediction=False):
        import pdb
        pdb.set_trace()
        obs = np.array(obs).astype(np.float32)
        obs = obs / 255.0

        if prediction:
            obs = np.expand_dims(obs, 0)

        N, S, C, W, H = obs.shape
        obs = obs.reshape(N, S*C, H, W)
        obs = torch.FloatTensor(obs).to(self.device)

        return obs
