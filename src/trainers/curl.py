from .base import BaseTrainer
from src.common.losses import CURLLoss
from einops import rearrange
import torch
import torch.nn.functional as F
import copy

class CURLTrainer(BaseTrainer):
    name = 'curl'
    def __init__(self,
                 cfg,
                 device,
                 dataloader,
                 logger, 
                 aug_func,
                 model):
        
        super().__init__(cfg, device, dataloader, logger, aug_func, model)  
        self.target_model = copy.deepcopy(self.model).to(self.device)  

    def compute_loss(self, obs, act, rew, done):
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t c h w -> n (t c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x1 = rearrange(x1, 'n (t c) h w -> n t c h w', t=self.cfg.obs_shape[0])
        x2 = rearrange(x2, 'n (t c) h w -> n t c h w', t=self.cfg.obs_shape[0])
        x = torch.cat([x1, x2], axis=0)

        # online encoder
        y_o, _ = self.model.backbone(x)
        z_o = self.model.head.project(y_o)
        p_o = self.model.head.predict(z_o)
        p_o = rearrange(p_o, 'n t d -> (n t) d')
        p1_o, p2_o = p_o.chunk(2)

        # target encoder
        with torch.no_grad():
            y_t, _ = self.target_model.backbone(x)
            z_t = self.target_model.head.project(y_t)
            z_t = rearrange(z_t, 'n t d -> (n t) d')
            z1_t, z2_t = z_t.chunk(2)

        # loss
        loss_fn = CURLLoss(temperature=self.cfg.temperature)
        loss = 0.5 * (loss_fn(p1_o, z2_t) + loss_fn(p2_o, z1_t))
        loss = torch.mean(loss)

        # logs
        log_data = {'loss': loss.item()}
        
        return loss, log_data
    
    def update(self, obs, act, rew, done):
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = self.cfg.tau * target.data + (1 - self.cfg.tau) * online.data
    
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