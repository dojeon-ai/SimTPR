from .base import BaseAgent
from src.common.train_utils import LinearScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import tqdm


class BYOLTrainer(BaseTrainer):
    name = 'dqn'
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

    def _update(self):
        idxes, obs_batch, act_batch, rew_batch, done_batch, next_obs_batch, weights = self.buffer.sample(self.cfg.batch_size)
        
        # augment the observation if needed
        obs_batch, next_obs_batch = self.aug_func(obs_batch), self.aug_func(next_obs_batch)
        
        # y^ = Q_{theta}(s,a)
        cur_online_q = self.model(obs_batch)
        pred = cur_online_q.gather(1, act_batch.unsqueeze(-1)).flatten()

        # y = r + gamma * Q_{phi}(s',a')
        # standard Q: a'=arg_max_a Q_{phi}(s', a)
        # double Q: a'=arg max_a Q_{theta}(s', a)
        with torch.no_grad():
            next_target_q = self.target_model(next_obs_batch)
            if self.cfg.double:
                next_online_q = self.model(next_obs_batch) 
                next_act = torch.argmax(next_online_q, 1)
            else:            
                next_act = torch.argmax(next_target_q, 1)

            target_q = next_target_q.gather(1, next_act.unsqueeze(-1)).flatten()
            # Q = R_n + (γ^n)Q (w/ n-step return)
            target = rew_batch + (self.cfg.gamma ** self.buffer.n_step) * target_q * (1-done_batch)

        huber_distance = F.smooth_l1_loss(pred, target, reduction='none')
        loss = (huber_distance * weights).mean()

        # update priority 
        if self.buffer.name == 'per_buffer':
            self.buffer.update_priorities(idxs=idxs, priorities=huber_distance.detach().cpu().numpy())

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
        self.optimizer.step()

        # log
        log_data = {
            'loss': loss.item()
        }
        return log_data

    def train(self):
        self.model.train()
        self.target_model.train()
        obs = self.train_env.reset()
        for t in range(self.cfg.num_timesteps):
            # encode last observation to torch.tensor()
            obs_tensor = self.buffer.encode_obs(obs, prediction=True)

            # get action from the model
            eps = self.epsilon_scheduler.get_value(step=t)
            action = self.predict(obs_tensor, eps)

            # step
            next_obs, reward, done, info = self.train_env.step(action)

            # store new transition
            self.buffer.store(obs, action, reward, done, next_obs)

            # update
            if (t >= self.cfg.min_buffer_size) & (t % self.cfg.update_freq == 0):
                for _ in range(self.cfg.updates_per_step):
                    log_data = self._update()
                    self.logger.update_log(mode='train', **log_data)

            if t % self.cfg.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # logger
            self.logger.step(obs, reward, done, info, mode='train')
            if t % self.cfg.log_every == 0:
                self.logger.write_log()

            # move on
            # should not be done (cannot collect the return of trajectory)
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

                # get action from the model
                with torch.no_grad():
                    action = self.predict(obs_tensor, eps=0.01)

                # step
                next_obs, reward, done, info = self.eval_env.step(action)

                # logger
                self.logger.step(obs, reward, done, info, mode='eval')

                # move on
                if info.traj_done:
                    self.logger.write_log(mode='eval')
                    break
                else:
                    obs = next_obs