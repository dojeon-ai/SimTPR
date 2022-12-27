from .base import BaseTrainer
from src.common.losses import BarlowLoss
import torch
import torch.nn.functional as F
from einops import rearrange


class RSSMTrainer(BaseTrainer):
    name = 'rssm'
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

    def compute_loss(self, obs, act, rew, done, rtg, mode):
        n, t, f, c, h, w = obs.shape
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)        
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f)

        # encoder
        y, _ = self.model.backbone(x)

        kl_loss, states, rnn_hiddens = self.model.head.kl_loss(y)

        states = rearrange(states, 't n d -> (t n) d')
        rnn_hiddens = rearrange(rnn_hiddens, 't n d -> (t n) d')
        recon_loss = self.model.head.recon_loss(obs / 255.0, states, rnn_hiddens)

        loss = kl_loss + recon_loss
        
        log_data = {'loss': loss.item(),
                    'obs_loss': recon_loss.item()}
        
        return loss, log_data