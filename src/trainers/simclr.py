from .base import BaseTrainer
from src.common.train_utils import LinearScheduler
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
        self.target_model = copy.deepcopy(self.model).to(self.device)        
        self.optimizer = self._build_optimizer(cfg.optimizer)

    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError

    def _update_moving_average(self):
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = self.cfg.tau * target.data + (1 - self.cfg.tau) * online.data

    def _compute_loss(self, obs, done):
        # obs
        T, B, S, C, H, W = obs.shape
        obs = obs.reshape(T*B, S*C, H, W) 
        obs = obs.float() / 255.0
        obs1, obs2 = self.aug_func(obs), self.aug_func(obs)
        obs = torch.cat([obs1, obs2], axis=0)

        # online encoder
        y_o = self.model.backbone(cur_obs)
        z_o = self.model.head.project(y_o)
        z1_o, z2_o = z_o.chunk(2)

        # target encoder
        with torch.no_grad():
            y_t = self.target_model.backbone(obs)
            z_t = self.target_model.head.project(y_t)
            z1_t, z2_t = z_t.chunk(2)

        # TODO: (1) change loss (2) reflect dones
        import pdb
        pdb.set_trace()


        return loss

    def train(self):
        self.model.train()
        self.target_model.train()
        t = 0
        for e in range(self.cfg.num_epochs):
            for batch in tqdm.tqdm(self.dataloader):
                # forward
                obs = batch.observation.to(self.device)
                done = batch.done.to(self.device)
                loss = self._get_loss(obs, done)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._update_moving_average()

                # log
                log_data = {'loss': loss.item()}
                self.logger.update_log(**log_data)
                if t % self.cfg.log_every == 0:
                    self.logger.write_log()

                # proceed
                t += 1

    def evaluate(self):
        self.model.eval()
        pass