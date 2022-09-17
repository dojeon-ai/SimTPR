from .base import BaseTrainer
from src.common.losses import ContrastiveLoss
import torch
import torch.nn.functional as F
from einops import rearrange


class SimCLRTrainer(BaseTrainer):
    name = 'simclr'
    def __init__(self,
                 cfg,
                 device,
                 dataloader,
                 logger, 
                 aug_func,
                 model):
        
        super().__init__(cfg, device, dataloader, logger, aug_func, model)  

    def compute_loss(self, obs, act, rew, done):
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t c h w -> n (t c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x1 = rearrange(x1, 'n (t c) h w -> n t c h w', t=self.cfg.obs_shape[0])
        x2 = rearrange(x2, 'n (t c) h w -> n t c h w', t=self.cfg.obs_shape[0])
        x = torch.cat([x1, x2], axis=0)

        # encoder
        y, _ = self.model.backbone(x)
        z, _ = self.model.head(y)
        z = rearrange(z, 'n t d -> (n t) d')

        # loss
        loss_fn = ContrastiveLoss(temperature=self.cfg.temperature)
        loss = loss_fn(z)
        
        # logs
        log_data = {'loss': loss.item()}
        
        return loss, log_data

    def evaluate(self, obs, act, rew, done):
        self.model.eval()
        
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t c h w -> n (t c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x1 = rearrange(x1, 'n (t c) h w -> n t c h w', t=self.cfg.obs_shape[0])
        x2 = rearrange(x2, 'n (t c) h w -> n t c h w', t=self.cfg.obs_shape[0])
        x = torch.cat([x1, x2], axis=0)
        
        # encoder
        y, _ = self.model.backbone(x)
        z = self.model.head.project(y)
        z = rearrange(z, 'n t d -> (n t) d')
        z1, z2 = z.chunk(2)
        
        # similarity
        pos_idx = torch.eye(z1.shape[0], device=z1.device)
        sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        pos_sim = (torch.sum(sim * pos_idx) / torch.sum(pos_idx)).item()
        neg_sim = (torch.sum(sim * (1-pos_idx)) / torch.sum(1-pos_idx)).item()
        pos_neg_diff = pos_sim - neg_sim
        
        # logs
        log_data = {'pos_sim': pos_sim,
                    'neg_sim': neg_sim,
                    'pos_neg_diff': pos_neg_diff} 
        
        return log_data