from .base import BaseTrainer
from src.common.losses import BarlowLoss
import torch
import torch.nn.functional as F
from einops import rearrange
import wandb


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

        ##############
        # encode
        y, _ = self.model.backbone(x)
        
        ###############
        # compute loss
        # kl loss
        kl_loss, states, rnn_hiddens = self.model.head.kl_loss(y)
        # recon loss
        states = rearrange(states, 't n d -> (n t) d')
        rnn_hiddens = rearrange(rnn_hiddens, 't n d -> (n t) d')
        obs_target = obs / 255.0
        recon_loss, recon, obs_target = self.model.head.recon_loss(obs_target, states, rnn_hiddens)

        # loss
        kl_lmbda = self.cfg.kl_lmbda
        loss = recon_loss + kl_lmbda * kl_loss
        
        ################
        # log
        log_data = {'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'recon_image': wandb.Image(recon),
                    'obs_image': wandb.Image(obs_target)}
        
        return loss, log_data