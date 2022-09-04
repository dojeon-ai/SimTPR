from .base import BaseTrainer
from src.common.train_utils import LinearScheduler, CosineAnnealingWarmupRestarts
from src.common.vis_utils import rollout_attn_maps
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
from fractions import Fraction
import random
import copy
import tqdm


class VQVAETrainer(BaseTrainer):
    name = 'vq_vae'
    def __init__(self,
                 cfg,
                 device,
                 dataloader,
                 logger, 
                 aug_func,
                 model):
        
        super().__init__()  
        self.cfg = cfg  
        self.device = device
        self.dataloader = dataloader
        self.logger = logger

        self.aug_func = aug_func.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = self._build_optimizer(cfg.optimizer)
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.num_epochs, cfg.scheduler)
        
    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, num_epochs, scheduler_cfg):
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             first_cycle_steps=num_epochs,
                                             **scheduler_cfg)
 

    def compute_loss(self, obs, act, done):        
        x = obs / 255.0
        x = rearrange(x, 'n t s c h w -> n (t s c) h w')
        
        # encoder
        x = self.model.backbone(x)
        self.model.head(x)
        # codebook
        
        
        import pdb
        pdb.set_trace()
        
        # perform augmentation if needed
        
        
        #x = rearrange(obs, 'n t s c h w -> n (t c) h w')
        
            
        return loss, log_data
        
    def train(self):
        self.model.train()
        loss, t = 0, 0
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.dataloader):                         
                # forward
                obs = batch.observation.to(self.device)
                act = batch.action.to(self.device)
                done = batch.done.to(self.device)
                loss, log_data = self.compute_loss(obs, act, done)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm.append(p.grad.detach().data.norm(2))
                grad_norm = torch.stack(grad_norm)
                log_data['min_grad_norm'] = torch.min(grad_norm).item()
                log_data['mean_grad_norm'] = torch.mean(grad_norm).item()
                log_data['max_grad_norm'] = torch.max(grad_norm).item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()

                # evaluation
                if t % self.cfg.eval_every == 0:
                    pass
                    
                # log
                self.logger.update_log(**log_data)
                if t % self.cfg.log_every == 0:
                    self.logger.write_log()

                # proceed
                t += 1
            if e % self.cfg.save_every == 0:
                self.logger.save_state_dict(model=self.model, epoch=e)

            self.lr_scheduler.step()
