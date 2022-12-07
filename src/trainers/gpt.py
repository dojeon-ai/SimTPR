from .base import BaseTrainer
from src.common.losses import ConsistencyLoss, CURLLoss, BarlowLoss
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

class GPTTrainer(BaseTrainer):
    name = 'gpt'
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
        max_rtg = 0
        datasets = self.train_loader.dataset.datasets
        for dataset in datasets:
            max_rtg = max(max_rtg, torch.max(dataset.rtg).item())
        self.max_rtg = max_rtg
        
        
    def compute_loss(self, obs, act, rew, done, rtg):
        ####################
        # augmentation
        n, t, f, c, h, w = obs.shape
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x1, x2 = self.aug_func(x), self.aug_func(x)
        x = torch.cat([x1, x2], axis=0)        
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f)
        act = torch.cat([act, act], axis=0)
        rew = torch.cat([rew, rew], axis=0)
        rtg = torch.cat([rtg, rtg], axis=0) / self.max_rtg
        
        #################
        # forward
        
        # encode
        y, _ = self.model.backbone(x)  
        z = self.model.head.encode_obs(y)
        
        z_o = z[:, :-1]
        z_t = z[:, 1:]
        act = act[:, :-1]
        rew = rew[:, :-1]
        rtg = rtg[:, :-1]
        
        # mask based on dataset-type
        d_type = self.cfg.dataset_type 
        n, t = act.shape
        
        if d_type == 'video':
            act_mask = torch.ones((n, t), device=x.device)
            rew_mask = torch.ones((n, t), device=x.device)
            rtg_mask = torch.ones((n, t), device=x.device)
        
        elif d_type == 'demonstration':
            act_mask = torch.zeros((n, t), device=x.device)
            rew_mask = torch.ones((n, t), device=x.device)
            rtg_mask = torch.ones((n, t), device=x.device)
        
        elif d_type == 'trajectory':
            act_mask = torch.zeros((n, t), device=x.device)
            rew_mask = torch.zeros((n, t), device=x.device)
            rtg_mask = torch.zeros((n, t), device=x.device)
        
        # decode
        obs_o, act_o, rew_o, rtg_o = self.model.head.decode(z_o, act, rew, rtg,
                                                            act_mask, rew_mask, rtg_mask)  
        obs_p, act_p, rew_p, rtg_p = self.model.head.predict(obs_o, act_o, rew_o, rtg_o) 

        #################
        # loss
        
        # obs loss            
        obs_p1, obs_p2 = obs_p.chunk(2)
        obs_p1, obs_p2 = rearrange(obs_p1, 'n t d -> (n t) d'), rearrange(obs_p2, 'n t d -> (n t) d')

        obs_z = z_t
        obs_z1, obs_z2 = obs_z.chunk(2)
        obs_z1, obs_z2 = rearrange(obs_z1, 'n t d -> (n t) d'), rearrange(obs_z2, 'n t d -> (n t) d')

        cons_loss_fn = ConsistencyLoss()       
        cons_loss = 0.5 * (cons_loss_fn(obs_p1, obs_z2.detach()) + cons_loss_fn(obs_p2, obs_z1.detach()))
        cons_loss = torch.mean(cons_loss)
        
        barlow_loss_fn = BarlowLoss(self.cfg.lmbda)
        barlow_loss = 0.5 * (barlow_loss_fn(obs_z1, obs_z2) + barlow_loss_fn(obs_z2, obs_z1))
            
        obs_loss = self.cfg.cons_lmbda * cons_loss + self.cfg.barlow_lmbda * barlow_loss
        
            
        # act loss
        act_loss_fn = nn.CrossEntropyLoss(reduction='none')
        act_p = rearrange(act_p, 'n t d -> (n t) d')
        act_t = rearrange(act, 'n t -> (n t)')
        act_loss = act_loss_fn(act_p, act_t)
        
        act_idx = (1-act_mask).flatten()
        act_loss = torch.sum(act_loss * act_idx) / (torch.sum(act_idx) + 1e-6)
        act_acc = torch.mean((torch.argmax(act_p, 1) == act_t).float())
  
        # rew loss
        rew_loss_fn = nn.MSELoss(reduction='none')        
        rew_p = rearrange(rew_p, 'n t 1 -> (n t 1)')
        rew = rearrange(rew, 'n t -> (n t)')
        rew_loss = rew_loss_fn(rew_p, rew)

        rew_idx = (1-rew_mask).flatten()
        rew_loss = torch.sum(rew_loss * rew_idx) / (torch.sum(rew_idx) + 1e-6)
            
        # rtg loss
        rtg_loss_fn = nn.MSELoss(reduction='none')
        rtg_p = rearrange(rtg_p, 'n t 1 -> (n t 1)')
        rtg = rearrange(rtg, 'n t -> (n t)')
        rtg_loss = rtg_loss_fn(rtg_p, rtg)

        rtg_idx = (1-rtg_mask).flatten()
        rtg_loss = torch.sum(rtg_loss * rtg_idx) / (torch.sum(rtg_idx) + 1e-6)
        
        loss = (self.cfg.obs_lmbda * obs_loss + 
                self.cfg.act_lmbda * act_loss + 
                self.cfg.rew_lmbda * rew_loss + 
                self.cfg.rtg_lmbda * rtg_loss)
        
        ###############
        # logs
        # quantitative
        pos_idx = torch.eye(obs_p1.shape[0], device=x.device)
        sim = F.cosine_similarity(obs_p1.unsqueeze(1), obs_z2.unsqueeze(0), dim=-1)
        pos_sim = (torch.sum(sim * pos_idx) / torch.sum(pos_idx))
        neg_sim = (torch.sum(sim * (1-pos_idx)) / torch.sum(1-pos_idx))
        pos_neg_diff = pos_sim - neg_sim
        
        log_data = {'loss': loss.item(),
                    'obs_loss': obs_loss.item(),
                    'barlow_loss': barlow_loss.item(),
                    'cons_loss': cons_loss.item(),
                    'act_loss': act_loss.item(),
                    'rew_loss': rew_loss.item(),
                    'rtg_loss': rtg_loss.item(),
                    'act_acc': act_acc.item(),
                    'pos_sim': pos_sim.item(),
                    'neg_sim': neg_sim.item(),
                    'pos_neg_diff': pos_neg_diff.item()}
        
        return loss, log_data

    
    def update(self, obs, act, rew, done, rtg):
        pass

            
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
            obs = torch.FloatTensor(obs).to(self.device)
            obs = rearrange(obs, 'n f c h w -> n 1 f c h w')
            with torch.no_grad():
                obs, _ = self.model.backbone(obs)
                obs = self.model.head.obs_to_latent(obs)
                obs = obs.cpu()

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

            