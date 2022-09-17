from abc import *
from typing import Tuple
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from src.common.train_utils import CosineAnnealingWarmupRestarts, get_grad_norm_stats


class BaseTrainer():
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
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.scheduler)
        
    @classmethod
    def get_name(cls):
        return cls.name

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

    def _build_scheduler(self, optimizer, scheduler_cfg):
        first_cycle_steps = len(self.dataloader) * self.cfg.num_epochs
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             first_cycle_steps=first_cycle_steps,
                                             **scheduler_cfg)
 
    @abstractmethod
    def compute_loss(self, obs, act, rew, done) -> Tuple[torch.Tensor, dict]:
        pass
    
    @abstractmethod
    # custom model update other than backpropagation (e.g., ema)
    def update(self, obs, act, rew, done):
        pass
    
    def train(self):
        step = 0
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.dataloader):   
                # forward
                self.model.train()
                obs = batch.observation.to(self.device)
                act = batch.action.to(self.device)
                rew = batch.reward.to(self.device)
                done = batch.done.to(self.device)
                loss, train_logs = self.compute_loss(obs, act, rew, done)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                grad_stats = get_grad_norm_stats(self.model)
                train_logs.update(grad_stats)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()
                self.update(obs, act, rew, done)

                # evaluate
                if step % self.cfg.eval_every == 0:
                    self.model.eval()
                    eval_logs = self.evaluate(obs, act, rew, done)
                    
                # log
                log_data = {}
                log_data.update(train_logs)
                log_data.update(eval_logs)                
                self.logger.update_log(**log_data)
                if step % self.cfg.log_every == 0:
                    self.logger.write_log(step)
                    
                # proceed
                self.lr_scheduler.step()
                step += 1
                
            if e % self.cfg.save_every == 0:
                self.logger.save_state_dict(model=self.model, epoch=e)

    @abstractmethod
    def evaluate(self, obs, act, rew, done) -> dict:
        pass
    
