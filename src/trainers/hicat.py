from .base import BaseTrainer
from src.common.losses import ConsistencyLoss, CURLLoss
from src.common.train_utils import LinearScheduler
from src.common.vit_utils import get_random_3d_mask, get_3d_masked_input, restore_masked_input
from src.common.beam import Beam
from einops import rearrange
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import tqdm
import random
import wandb

class HiCaTTrainer(BaseTrainer):
    name = 'hicat'
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 eval_act_loader,
                 eval_rew_loader,
                 env,
                 logger, 
                 agent_logger,
                 aug_func,
                 model):
        
        super().__init__(cfg, device, 
                         train_loader, eval_act_loader, eval_rew_loader, env,
                         logger, agent_logger, aug_func, model)  
        self.target_model = copy.deepcopy(self.model).to(self.device)        
        update_steps = len(self.train_loader) * self.cfg.num_epochs
        cfg.tau_scheduler.step_size = update_steps
        self.tau_scheduler = LinearScheduler(**cfg.tau_scheduler)
        
        max_rtg = 0
        datasets = self.train_loader.dataset.datasets
        for dataset in datasets:
            max_rtg = max(max_rtg, torch.max(dataset.rtg).item())
        self.max_rtg = max_rtg
        
        
    def compute_loss(self, obs, act, rew, done, rtg):
        ####################
        # augmentation
        n, t, f, c, h, w = obs.shape
        
        # weak augmentation to both online & target
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x = torch.cat([x1, x2], axis=0)        
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f)
        act = torch.cat([act, act], axis=0)
        rew = torch.cat([rew, rew], axis=0)
        rtg = torch.cat([rtg, rtg], axis=0) / self.max_rtg

        # strong augmentation to online: (masking)
        assert self.cfg.mask_type in {'none', 'pixel'}
        if self.cfg.mask_type == 'none':
            pass
        
        elif self.cfg.mask_type == 'pixel':
            ph, pw = self.cfg.patch_size[0], self.cfg.patch_size[1]
            nh, nw = (h // ph), (w//pw)
            np = nh * nw
            
            # generate random-mask
            video_shape = (2*n, t, np)
            ids_keep, _, ids_restore = get_random_3d_mask(video_shape, 
                                                          self.cfg.mask_ratio,
                                                          self.cfg.mask_strategy)
            ids_keep = ids_keep.to(x.device)
            ids_restore = ids_restore.to(x.device)
            
            # mask & restore
            x_o = rearrange(x, 'n t f c (nh ph) (nw pw) -> n t (nh nw) (ph pw f c)', ph=ph, pw=pw)
            x_o = get_3d_masked_input(x_o, ids_keep, self.cfg.mask_strategy)
            x_o = restore_masked_input(x_o, ids_restore)
            x_o = rearrange(x_o, 'n (t nh nw) (ph pw f c) -> n t f c (nh ph) (nw pw)', 
                            t=t, f=f, c=c, nh=nh, nw=nw, ph=ph, pw=pw)
            
        x_o = x[:, :-1]
        x_t = x[:, 1:]
        act = act[:, :-1]
        rew = rew[:, :-1]
        rtg = rtg[:, :-1]
        
        #########################
        # Online
        
        # Encoder
        y_o, _ = self.model.backbone(x_o)  
        
        # State Transformer
        z_o = self.model.head.encode_obs(y_o)
        s_obs_o = self.model.head.decode_state(z_o)
        
        # Demonstration Transformer
        d_obs_o = torch.cat((z_o[:, 0:1], s_obs_o[:, :-1]), 1)
        d_obs_o, d_act_o = self.model.head.decode_demonstration(d_obs_o, act)
        
        # Trajectory Transformer
        t_obs_o = torch.cat((z_o[:, 0:1], d_obs_o[:, :-1]), 1)
        t_obs_o, t_act_o, t_rew_o, t_rtg_o = \
            self.model.head.decode_trajectory(t_obs_o, d_act_o, rew, rtg)
        
        # Prediction
        s_obs_p_o = self.model.head.predict_state(s_obs_o)
        d_obs_p_o, d_act_p_o = self.model.head.predict_demonstration(d_obs_o, d_act_o)
        t_obs_p_o, t_act_p_o, t_rew_p_o, t_rtg_p_o = \
            self.model.head.predict_trajectory(t_obs_o, t_act_o, t_rew_o, t_rtg_o)
        
        s_obs_p1_o, s_obs_p2_o = s_obs_p_o.chunk(2)
        d_obs_p1_o, d_obs_p2_o = d_obs_p_o.chunk(2)
        t_obs_p1_o, t_obs_p2_o = t_obs_p_o.chunk(2)
        
        ##############################
        # Target
        with torch.no_grad():
            # Encoder
            y_t, _ = self.target_model.backbone(x_t)
            z_t = self.target_model.head.encode_obs(y_t)
            
            # Target
            s_obs_z_t = self.target_model.head.project_state(z_t)
            d_obs_z_t = self.target_model.head.project_demonstration(z_t)
            t_obs_z_t = z_t
        
            s_obs_z1_t, s_obs_z2_t = s_obs_z_t.chunk(2)
            d_obs_z1_t, d_obs_z2_t = d_obs_z_t.chunk(2)
            t_obs_z1_t, t_obs_z2_t = t_obs_z_t.chunk(2)
            
        #################
        # Loss
        obs_loss_fn = ConsistencyLoss()
        act_loss_fn = nn.CrossEntropyLoss()
        rew_loss_fn = nn.MSELoss()
        
        # State Loss
        # obs loss
        s_obs_p1_o = rearrange(s_obs_p1_o, 'n t d -> (n t) d')
        s_obs_p2_o = rearrange(s_obs_p2_o, 'n t d -> (n t) d')
        s_obs_z1_t = rearrange(s_obs_z1_t, 'n t d -> (n t) d')
        s_obs_z2_t = rearrange(s_obs_z2_t, 'n t d -> (n t) d')

        s_obs_loss = 0.5 * (obs_loss_fn(s_obs_p1_o, s_obs_z2_t) 
                          + obs_loss_fn(s_obs_p2_o, s_obs_z1_t))
        s_obs_loss = torch.mean(s_obs_loss)
        
        state_loss = s_obs_loss
        
        # Demonstration Loss
        # obs loss
        d_obs_p1_o = rearrange(d_obs_p1_o, 'n t d -> (n t) d')
        d_obs_p2_o = rearrange(d_obs_p2_o, 'n t d -> (n t) d')
        d_obs_z1_t = rearrange(d_obs_z1_t, 'n t d -> (n t) d')
        d_obs_z2_t = rearrange(d_obs_z2_t, 'n t d -> (n t) d')

        d_obs_loss = 0.5 * (obs_loss_fn(d_obs_p1_o, d_obs_z2_t) 
                          + obs_loss_fn(d_obs_p2_o, d_obs_z1_t))
        d_obs_loss = torch.mean(d_obs_loss)
        
        # act loss
        d_act_p_o = rearrange(d_act_p_o, 'n t d -> (n t) d')
        d_act_t = rearrange(act, 'n t -> (n t)')
        d_act_loss = act_loss_fn(d_act_p_o, d_act_t)
        d_act_acc = torch.mean((torch.argmax(d_act_p_o, 1) == d_act_t).float())
        
        demon_loss = d_obs_loss + d_act_loss
        
        # Trajectory Loss
        # obs loss
        t_obs_p1_o = rearrange(t_obs_p1_o, 'n t d -> (n t) d')
        t_obs_p2_o = rearrange(t_obs_p2_o, 'n t d -> (n t) d')
        t_obs_z1_t = rearrange(t_obs_z1_t, 'n t d -> (n t) d')
        t_obs_z2_t = rearrange(t_obs_z2_t, 'n t d -> (n t) d')

        t_obs_loss = 0.5 * (obs_loss_fn(t_obs_p1_o, t_obs_z2_t) 
                          + obs_loss_fn(t_obs_p2_o, t_obs_z1_t))
        t_obs_loss = torch.mean(t_obs_loss)
        
        # act loss
        t_act_p_o = rearrange(t_act_p_o, 'n t d -> (n t) d')
        t_act_t = rearrange(act, 'n t -> (n t)')
        t_act_loss = act_loss_fn(t_act_p_o, t_act_t)
        t_act_acc = torch.mean((torch.argmax(t_act_p_o, 1) == t_act_t).float())
        
        # rew loss
        t_rew_p_o = rearrange(t_rew_p_o, 'n t 1 -> (n t 1)')
        t_rew_t = rearrange(rew, 'n t -> (n t)')
        t_rew_loss = rew_loss_fn(t_rew_p_o, t_rew_t)
        
        # rtg loss
        t_rtg_p_o = rearrange(t_rtg_p_o, 'n t 1 -> (n t 1)')
        t_rtg_t = rearrange(rtg, 'n t -> (n t)')
        t_rtg_loss = rew_loss_fn(t_rtg_p_o, t_rtg_t)
        
        traj_loss = t_obs_loss + t_act_loss + t_rew_loss + t_rtg_loss

        loss = (self.cfg.state_lmbda * state_loss + 
                self.cfg.demon_lmbda * demon_loss + 
                self.cfg.traj_lmbda  * traj_loss)
        
        ###############
        # logs
        log_data = {'loss': loss.item(),
                    's_obs_loss': s_obs_loss.item(),
                    'd_obs_loss': d_obs_loss.item(),
                    'd_act_loss': d_act_loss.item(),
                    'd_act_acc': d_act_acc.item(),
                    't_obs_loss': t_obs_loss.item(),
                    't_act_loss': t_act_loss.item(),
                    't_rew_loss': t_rew_loss.item(),
                    't_rtg_loss': t_rtg_loss.item(),
                    't_act_acc': t_act_acc.item()}
        
        return loss, log_data

    
    def update(self, obs, act, rew, done, rtg):
        tau = self.tau_scheduler.get_value()
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

            
    def evaluate_policy(self):
        self.model.eval()
        
        # initialize history
        C = self.cfg.rollout.context_len
        obs_list = deque(maxlen=C)
        act_list = deque(maxlen=C) 
        rew_list = deque(maxlen=C)
        rtg_list = deque(maxlen=C)

        # initialize tree
        rollout_cfg = self.cfg.rollout
        rollout_type = rollout_cfg.pop('type')

        if rollout_type == 'mcts':        
            raise NotImplemented

        elif rollout_type == 'beam':
            tree = Beam(model=self.model, 
                        device=self.device,
                        num_envs=self.cfg.num_envs,
                        action_size=self.cfg.action_size,
                        rtg_scale=self.max_rtg,
                        **rollout_cfg)

        # run trajectory
        obs = self.env.reset()
        rtg = np.array([self.max_rtg] * self.cfg.num_envs)
        while True:
            # encode obs
            obs = np.array(obs).astype(np.float32)
            obs = obs / 255.0
            obs_list.append(obs)

            # select action with rollout
            state = {
                'obs_list': obs_list,
                'act_list': act_list,
                'rew_list': rew_list,
                'rtg_list': rtg_list
            }

            beam = tree.rollout(state)            
            tree_action = beam['act_batch'].numpy()[:, 0]
            rand_action = np.random.randint(0, self.cfg.action_size-1, self.cfg.num_envs)

            # eps-greedy
            eps = self.cfg.eval_eps
            prob = np.random.rand(self.cfg.num_envs)
            rand_idx = (prob <= eps)
            action = rand_idx * rand_action + (1-rand_idx) * tree_action

            # step
            next_obs, reward, done, info = self.env.step(action)

            # logger
            self.agent_logger.step(obs, reward, done, info)

            # move on
            if np.sum(self.agent_logger.traj_done) == self.cfg.num_envs:
                break
            else:
                obs = next_obs
                act_list.append(action)
                rew_list.append(reward)
                rtg_list.append(rtg / self.max_rtg)
                rtg = rtg - reward
    
        log_data = self.agent_logger.fetch_log()
        
        return log_data
