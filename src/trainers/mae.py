from .base import BaseTrainer
from src.common.train_utils import LinearScheduler
from src.common.train_utils import get_random_1d_mask, get_random_3d_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
import random
import copy
import tqdm


class MAETrainer(BaseTrainer):
    name = 'mae'
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
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.num_epochs)
        
        self.cfg.num_patches = self.model.backbone.num_patches

    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, num_epochs):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    def _compute_loss(self, obs, act, done):
        # reshape obs for augmentation
        obs = rearrange(obs, 'n t s c h w -> (n t) (s c) h w')
        obs = obs.float() / 255.0
        
        # perform augmentation if needed
        aug_obs = self.aug_func(obs)
        aug_obs = rearrange(aug_obs, '(n t) c h w -> n t c h w', n=self.cfg.batch_size, t=self.cfg.t_step) 

        # construct input data for vit
        x = {
            'img': aug_obs,
            'act': act,
            'done': done,
        }
        
        # construct mask data for vit-encoder
        video_shape = (self.cfg.batch_size, self.cfg.t_step, self.cfg.num_patches) 
        patch_ids_keep, patch_mask, patch_ids_restore = get_random_3d_mask(video_shape, self.cfg.patch_mask_ratio, self.cfg.patch_mask_type)
        
        act_shape = (self.cfg.batch_size, self.cfg.t_step) 
        act_ids_keep, act_mask, act_ids_restore = get_random_1d_mask(act_shape, self.cfg.act_mask_ratio)
        
        mask = {
            'patch_mask_type': self.cfg.patch_mask_type,
            'patch_ids_keep': patch_ids_keep.to(self.device),
            'patch_ids_restore': patch_ids_restore.to(self.device),
            'act_ids_keep': act_ids_keep.to(self.device),
            'act_ids_restore': act_ids_restore.to(self.device),
        }
        
        # forward
        
        
        
        obs_pred = self.model.backbone(x, mask)        
        
        pass

    def train(self):
        self.model.train()
        loss, t = 0, 0
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.dataloader):
                log_data = {}
                
                # forward
                obs = batch.observation.to(self.device)
                act = batch.action.to(self.device)
                done = batch.done.to(self.device)
                loss = self._compute_loss(obs, act, done)
                
                # backward
                if (t % self.cfg.update_freq == 0) and (t>0):
                    loss /= self.cfg.update_freq
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    log_data['loss'] = loss.item()
                    loss = 0.0

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

    def evaluate(self):
        self.model.eval()
        pass