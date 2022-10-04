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
                 train_loader,
                 eval_act_loader,
                 eval_rew_loader,
                 logger, 
                 aug_func,
                 model):
        
        super().__init__(cfg, device, 
                         train_loader, eval_act_loader, eval_rew_loader,
                         logger, aug_func, model)  

    def compute_loss(self, obs, act, rew, done):
        n, t, f, c, h, w = obs.shape
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x1 = rearrange(x1, 'n (t f c) h w -> n t f c h w', t=t, f=f)
        x2 = rearrange(x2, 'n (t f c) h w -> n t f c h w', t=t, f=f)
        x = torch.cat([x1, x2], axis=0)

        # encoder
        y, _ = self.model.backbone(x)
        z, _ = self.model.head(y)
        z = rearrange(z, 'n t d -> (n t) d')

        # loss
        loss_fn = ContrastiveLoss(temperature=self.cfg.temperature)
        loss = loss_fn(z)
        
        ###############
        # logs
        z1, z2 = z.chunk(2)
        pos_idx = torch.eye(z1.shape[0], device=z1.device)
        sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        pos_sim = (torch.sum(sim * pos_idx) / torch.sum(pos_idx))
        neg_sim = (torch.sum(sim * (1-pos_idx)) / torch.sum(1-pos_idx))
        pos_neg_diff = pos_sim - neg_sim
        
        log_data = {'loss': loss.item(),
                    'pos_sim': pos_sim.item(),
                    'neg_sim': neg_sim.item(),
                    'pos_neg_diff': pos_neg_diff.item()}
        
        return loss, log_data