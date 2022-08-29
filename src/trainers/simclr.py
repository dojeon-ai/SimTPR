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


class SimCLRTrainer(BaseTrainer):
    name = 'simclr'
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

    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, num_epochs):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    def _compute_loss(self, obs, done):
        # obs
        N, T, S, C, H, W = obs.shape
        obs = obs.reshape(N*T, S*C, H, W) 
        obs = obs.float() / 255.0
        obs1, obs2 = self.aug_func(obs), self.aug_func(obs)
        obs = torch.cat([obs1, obs2], axis=0)

        # encoder
        y = self.model.backbone(obs)
        z = self.model.head.project(y)

        # loss
        loss_fn = TemporalContrastiveLoss(num_trajectory=N,
                                          t_step=T,  
                                          temperature=self.cfg.temperature, 
                                          device=self.device)
        loss = loss_fn(z, done)
        return loss
    
    def _compute_similarity(self, obs, done):
        # obs
        N, T, S, C, H, W = obs.shape
        obs = obs.reshape(N*T, S*C, H, W) 
        obs = obs.float() / 255.0
        obs1, obs2 = self.aug_func(obs), self.aug_func(obs)
        obs = torch.cat([obs1, obs2], axis=0)

        # encoder
        y = self.model.backbone(obs)
        z = self.model.head.project(y)

        # loss
        sim_fn = TemporalSimilarityLoss(num_trajectory=N,
                                        t_step=T,  
                                        device=self.device)
        positive_sim, negative_sim = sim_fn(z, done)
        return positive_sim, negative_sim

    def train(self):
        self.model.train()
        loss, t = 0, 0
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.dataloader):
                log_data = {}

                # forward
                obs = batch.observation.to(self.device)
                done = batch.done.to(self.device)
                loss += self._compute_loss(obs, done)
                
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
                    with torch.no_grad():
                        positive_sim, negative_sim = self._compute_similarity(obs, done)
                    log_data['positive_sim'] = positive_sim.item()
                    log_data['negative_sim'] = negative_sim.item()
                    log_data['pos_neg_diff'] = positive_sim.item() - negative_sim.item()
                    
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