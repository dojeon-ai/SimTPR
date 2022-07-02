from .base import BaseAgent
from src.common.train_utils import LinearScheduler, RMS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import tqdm
import numpy as np
import wandb
import os
from collections import deque


class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(self.device)
            ) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(
                    self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        
        return reward


class APT(BaseAgent):
    name = 'apt'
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
        self.target_model = copy.deepcopy(self.model).to(self.device)   
        for param in self.target_model.parameters():
            param.requires_grad = False     
        self.optimizer = self._build_optimizer(cfg.optimizer)

        # distributional
        self.num_atoms = self.model.policy.get_num_atoms()
        self.v_min = self.cfg.v_min
        self.v_max = self.cfg.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        rms = RMS(device=device)
        self.pbe = PBE(rms=rms, device=device, **cfg.pbe)

    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), 
                                 **optimizer_cfg)
        else:
            raise ValueError

    def predict(self, obs):
        # [params] obs (torch.tensor) (N, C, H, W)
        q_value = (self.model(obs) * self.support.reshape(1,1,-1)).sum(-1)
        action = torch.argmax(q_value, 1).item()

        return action

    # Acts with an ε-greedy policy (used for evaluation only)
    def predict_greedy(self, obs, eps):
        # [params] obs (torch.tensor) (N, C, H, W)
        # epsilon-greedy prediction
        p = random.random()
        if p < eps:
            action = random.randint(0, self.cfg.action_size-1)
        else:
            action = self.predict(obs)

        return action

    def _update(self):
        idxs, obs_batch, act_batch, _, done_batch, next_obs_batch, weights = self.buffer.sample(self.cfg.batch_size)

        # augment the observation if needed
        obs_batch1, obs_batch2, next_obs_batch = self.aug_func(obs_batch), self.aug_func(obs_batch), self.aug_func(next_obs_batch)

        # reset noise
        self.model.policy.reset_noise()
        self.target_model.policy.reset_noise()

        # [contrastive loss] 
        # forward
        B = self.cfg.batch_size
        obs_batch_cat = torch.cat([obs_batch1, obs_batch2], dim=0)
        feat = self.model.backbone(obs_batch_cat)
        proj = self.model.header(feat)
        proj = F.normalize(proj, dim=1)

        # get label and sim matrix
        sim_matrix = torch.mm(proj, proj.t().contiguous())
        labels = torch.eye(B*2, device=self.device)
        labels = labels.to(self.device)

        # discard the main diagonal from both: labels & similarity [2*B, 2*B-1]
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

        # get cross-entropy logits & labels
        positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.cfg.temperature
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        ce_loss = F.cross_entropy(logits, labels)

        # Calculate current state's q-value distribution
        # cur_online_log_q_dist: (N, A, N_A = num_atoms)
        # log_pred_q_dist: (N, N_A)
        cur_online_feat = self.model.backbone(obs_batch1)
        cur_online_log_q_dist = self.model.policy(cur_online_feat, log=True)
        act_idx = act_batch.reshape(-1,1,1).repeat(1,1,self.num_atoms)
        log_pred_q_dist = cur_online_log_q_dist.gather(1, act_idx).squeeze(1)

        # apt-reward
        with torch.no_grad():
            rew_batch = self.pbe(cur_online_feat).flatten()

        with torch.no_grad():
            # Calculate n-th next state's q-value distribution
            # next_target_q_dist: (N, A, N_A)
            # target_q_dist: (N, N_A)
            next_target_q_dist = (self.target_model(next_obs_batch))
            if self.cfg.double:
                next_online_q_dist = (self.model(next_obs_batch))
                next_online_q =  (next_online_q_dist * self.support.reshape(1,1,-1)).sum(-1)
                next_act = torch.argmax(next_online_q, 1)
            else:       
                next_target_q =  (next_target_q_dist * self.support.reshape(1,1,-1)).sum(-1)     
                next_act = torch.argmax(next_target_q, 1)  
            next_act_idx = next_act.reshape(-1,1,1).repeat(1,1,self.num_atoms)
            target_q_dist = next_target_q_dist.gather(1, next_act_idx).squeeze(1)
        
            # C51 (https://arxiv.org/abs/1707.06887, Algorithm 1)
            # Compute the projection 
            # Tz = R_n + (γ^n)Z (w/ n-step return) (N, N_A)
            # N-step return is not utilized in APT
            Tz = rew_batch.unsqueeze(-1) + self.cfg.gamma * self.support.unsqueeze(0) * (1-done_batch).unsqueeze(-1)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            # L2-projection
            b = (Tz - self.v_min) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = torch.zeros((self.cfg.batch_size, self.num_atoms), device=self.device)
            for idx in range(self.cfg.batch_size):
                # += operation do not allow to add value to same index multiple times
                m[idx].index_add_(0, l[idx], target_q_dist[idx] * (u[idx] - b[idx]))
                m[idx].index_add_(0, u[idx], target_q_dist[idx] * (b[idx] - l[idx]))
            
        # kl-divergence 
        kl_div = -torch.sum(m * log_pred_q_dist, -1)
        rl_loss = (kl_div * weights).mean()

        # optimization
        self.optimizer.zero_grad()
        loss = ce_loss + rl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
        self.optimizer.step()
    
        # update priority 
        if self.buffer.name == 'per_buffer':
            EPS = 1e-5
            kl_div = torch.clamp(kl_div, EPS, 1/EPS) # nan-guard
            self.buffer.update_priorities(idxs=idxs, priorities=kl_div.detach().cpu().numpy())

        # log
        log_data = {
            'ce_loss': ce_loss.item(),
            'rl_loss': rl_loss.item(),
            'loss': loss.item(),
            'reward': rew_batch.mean().item()
        }
        return log_data

    def train(self):
        self.model.train()
        self.target_model.train()
        obs = self.train_env.reset()

        for t in tqdm.tqdm(range(1, self.cfg.num_timesteps+1)):
            # encode last observation to torch.tensor()
            obs_tensor = self.buffer.encode_obs(obs, prediction=True)

            # get action from the model
            self.model.policy.reset_noise()
            action = self.predict(obs_tensor)

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
            
            # evaluate & save model
            if t % self.cfg.eval_every == 0:
                self.logger.save_state_dict(model=self.model)
                self.evaluate()
                self.model.train()

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
                    action = self.predict_greedy(obs_tensor, eps=0.001)

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
