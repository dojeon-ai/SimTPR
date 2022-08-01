from .base import BaseTrainer
from src.common.train_utils import LinearScheduler
from src.common.losses import TemporalContrastiveLoss, TemporalSimilarityLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.update_epochs = cfg.num_epochs // cfg.time_span
        self.lr_scheduler = self._build_scheduler(self.optimizer, self.update_epochs)

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
        # obs
        N, T, S, C, H, W = obs.shape
        obs = obs.reshape(N*T, S*C, H, W) 
        obs = obs.float() / 255.0
        aug_obs = self.aug_func(obs)
        obs_pred = self.model(aug_obs, act, done)        
        
        pass

    def train(self):
        self.model.train()
        loss, t = 0, 0
        for u_e in range(1, self.update_epochs+1):
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
            
            n_epoch = u_e * self.cfg.time_span
            if n_epoch % self.cfg.save_every == 0:
                self.logger.save_state_dict(model=self.model, epoch=n_epoch)

            self.lr_scheduler.step()

    def evaluate(self):
        self.model.eval()
        pass