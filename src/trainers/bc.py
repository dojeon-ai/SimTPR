from .base import BaseTrainer
from src.common.losses import ConsistencyLoss
from src.common.train_utils import LinearScheduler
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BCTrainer(BaseTrainer):
    name = 'bc'
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
        ##############
        # forward
        n, t, f, c, h, w = obs.shape
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # online encoder
        y, _ = self.model.backbone(x)
        act_pred, _ = self.model.head(y)

        # loss
        loss_fn = nn.CrossEntropyLoss()
        act_pred = rearrange(act_pred, 'n t d -> (n t) d')
        act = rearrange(act, 'n t -> (n t)')
        loss = loss_fn(act_pred, act)
        act_acc = torch.mean((torch.argmax(act_pred, 1) == act).float())

        ###############
        # logs        
        log_data = {'loss': loss.item(),
                    'act_acc': act_acc.item()}
        
        return loss, log_data

    