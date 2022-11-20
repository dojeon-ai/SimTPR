from .base import BaseTrainer
from src.common.losses import ConsistencyLoss
from src.common.train_utils import LinearScheduler
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class ScratchTrainer(BaseTrainer):
    name = 'scratch'
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 eval_act_loader,
                 eval_rew_loader,
                 env,
                 logger, 
                 agent_logger,
                 aug_func,
                 model):
        
        super().__init__(cfg, device, 
                         train_loader, eval_act_loader, eval_rew_loader, env,
                         logger, agent_logger, aug_func, model)  

    def compute_loss(self, obs, act, rew, done, rtg):
        loss = torch.zeros(1, requires_grad=True)
        log_data = {}
        
        return loss, log_data

    def update(self, obs, act, rew, done, rtg):
        pass
    