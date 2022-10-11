import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from abc import *
from typing import Tuple


class BaseAgent(metaclass=ABCMeta):
    def __init__(self,
                 cfg,
                 device,
                 train_env,
                 eval_env,
                 logger, 
                 buffer,
                 aug_func,
                 model):
        
        super().__init__()  
        self.cfg = cfg  
        self.device = device
        self.train_env = train_env
        self.eval_env = eval_env
        self.logger = logger
        self.buffer = buffer
        self.aug_func = aug_func.to(self.device)
        self.model = model.to(self.device)
        
        finetune_type = cfg.pop('finetune_type')
        if finetune_type == 'naive':
            param_group = self.model.parameters()
        elif finetune_type == 'freeze':
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            param_group = self.model.parameters()
        
        self.optimizer = self._build_optimizer(param_group, cfg.optimizer)

    @classmethod
    def get_name(cls):
        return cls.name

    def _build_optimizer(self, param_group, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(param_group, 
                              **optimizer_cfg)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(param_group, 
                                 **optimizer_cfg)
        else:
            raise ValueError
    
    @abstractmethod
    def predict(self, obs, mode) -> torch.Tensor:
        pass
    
    @abstractmethod
    def compute_loss(self) -> Tuple[torch.Tensor, dict]:
        pass
    
    @abstractmethod
    # custom model update other than backpropagation (e.g., target model update)
    def update(self):
        pass
    
    def train(self):
        obs = self.train_env.reset()
        for t in tqdm.tqdm(range(1, self.cfg.num_timesteps+1)):
            # forward
            self.model.train()            
            obs_tensor = self.buffer.encode_obs(obs, prediction=True)
            action = self.predict(obs_tensor, mode='train')
            next_obs, reward, done, info = self.train_env.step(action)

            # store new transition
            self.buffer.store(obs, action, reward, done, next_obs)

            # optimize
            if (t >= self.cfg.min_buffer_size) & (t % self.cfg.optimize_freq == 0):
                for _ in range(self.cfg.optimize_per_step):
                    self.optimizer.zero_grad()
                    loss, log_data = self.compute_loss()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   self.cfg.clip_grad_norm)
                    self.optimizer.step()
                    self.logger.update_log(mode='train', **log_data)
                
            if (t >= self.cfg.min_buffer_size) & (t % self.cfg.update_freq == 0):
                self.update()

            # evaluate
            if t % self.cfg.eval_every == 0:
                self.evaluate()
            
            # log
            self.logger.step(obs, reward, done, info, mode='train')
            if t % self.cfg.log_every == 0:
                self.logger.write_log(mode='train')

            # reset when trajectory is done
            if info.traj_done:
                obs = self.train_env.reset()
            else:
                obs = next_obs

    def evaluate(self):
        self.model.eval()
        for _ in tqdm.tqdm(range(self.cfg.num_eval_trajectories)):
            obs = self.eval_env.reset()
            while True:
                # encode last observation to torch.tensor()
                obs_tensor = self.buffer.encode_obs(obs, prediction=True)

                # evaluation is based on greedy prediction
                with torch.no_grad():
                    action = self.predict(obs_tensor, mode='eval')

                # step
                next_obs, reward, done, info = self.eval_env.step(action)

                # logger
                self.logger.step(obs, reward, done, info, mode='eval')

                # move on
                if info.traj_done:
                    break
                else:
                    obs = next_obs
        
        self.logger.write_log(mode='eval')

