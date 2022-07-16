from .base import BaseTrainer
from src.common.train_utils import LinearScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import tqdm


class BYOLTrainer(BaseTrainer):
    name = 'byol'
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

    def _update_moving_average(self):
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = self.cfg.tau * target.data + (1 - self.cfg.tau) * online.data

    def _compute_loss(self, obs, done):
        # obs
        T, N, S, C, H, W = obs.shape
        obs = obs.reshape(T*N, S*C, H, W) 
        obs = obs.float() / 255.0
        obs1, obs2 = self.aug_func(obs), self.aug_func(obs)
        obs = torch.cat([obs1, obs2], axis=0)
        #cur_obs1, cur_obs2 = obs1.reshape(T, B, S*C, H, W)[0], obs2.reshape(T, B, S*C, H, W)[0]

        # online encoder
        y_o = self.model.backbone(obs)
        z_o = self.model.head.project(y_o)
        p_o = self.model.head.predict(z_o)
        p1_o, p2_o = p_o.chunk(2)

        # target encoder
        with torch.no_grad():
            y_t = self.target_model.backbone(obs)
            z_t = self.target_model.head.project(y_t)
            z1_t, z2_t = z_t.chunk(2)

        # loss
        loss_fn = TemporalConsistencyLoss(time_span=T, 
                                          num_trajectory=N, 
                                          device=self.device)
        loss = -0.5 * (loss_fn(p1_o, z2_t, done) + loss_fn(p2_o, z1_t, done))

        # TODO: (1) change loss (2) reflect dones
        # Temporal BYOL Loss
        #p1_o, p2_o = p1_o.repeat([T, 1]), p2_o.repeat([T, 1])
        #loss = -0.5 * (F.cosine_similarity(p1_o, z2_t.detach(), dim=-1).mean() 
        #                + F.cosine_similarity(p2_o, z1_t.detach(), dim=-1).mean()) 
        
        return loss

    def train(self):
        self.model.train()
        self.target_model.train()
        t = 0
        for e in range(self.update_epochs):
            for batch in tqdm.tqdm(self.dataloader):
                # forward
                obs = batch.observation.to(self.device)
                done = batch.done.to(self.device)
                loss = self._compute_loss(obs, done)
                
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
            
            if e % self.cfg.eval_every == 0:
                self.logger.save_state_dict(model=self.model, epoch=e)

            self.lr_scheduler.step()

    def evaluate(self):
        self.model.eval()
        pass