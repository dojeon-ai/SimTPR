from .base import BaseTrainer
from src.common.train_utils import LinearScheduler
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import tqdm
import numpy as np

class DTTrainer(BaseTrainer):
    name = 'dt'
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
        
    def compute_loss(self, obs, act, rew, done, rtg):
        ####################
        # augmentation
        n, t, f, c, h, w = obs.shape
        
        x = obs / 255.0
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        #################
        # forward
        # online encoder
        x, _ = self.model.backbone(x)  
        
        # decode
        dec_input = {'rtg': rtg,
                     'obs': x,
                     'act': act}
    
        act_pred = self.model.head.decode(dec_input)

        # loss
        loss_fn = nn.CrossEntropyLoss()
        act_pred = rearrange(act_pred, 'n t d -> (n t) d')
        act = rearrange(act, 'n t -> (n t)')
        loss = loss_fn(act_pred, act)
        act_acc = torch.mean((torch.argmax(act_pred, 1) == act).float())
        
        ###############
        # logs        
        log_data = {'loss': loss.item(),
                    'act_acc': act_acc.item()}
        
        return loss, log_data
    
    
    def evaluate_policy(self):
        self.model.eval()
        
        datasets = self.train_loader.dataset.datasets
        max_rtg = 0
        for dataset in datasets:
            max_rtg = max(max_rtg, torch.max(dataset.rtg).item())
        max_rtg = max_rtg * self.cfg.max_rtg_ratio
        
        for _ in tqdm.tqdm(range(self.cfg.num_eval_trajectories)):
            PAD = 0
            d = self.model.head.in_dim
            t = self.cfg.t_step
            rtg_list = [PAD] * self.cfg.t_step
            obs_list = [torch.zeros((1,1,d), device=self.device)] * self.cfg.t_step
            act_list = [PAD] * self.cfg.t_step
            
            rtg_list.append(max_rtg)
            obs = self.env.reset()
            while True:
                # encode obs
                # f,c,h,w -> 1, 1, f, c, h, w
                obs = np.array(obs).astype(np.float32)
                obs = obs / 255.0
                obs = np.expand_dims(obs, 0)
                obs = np.expand_dims(obs, 1)
                obs = torch.FloatTensor(obs).to(self.device)
                obs, _ = self.model.backbone(obs)
                obs_list.append(obs)

                # decode w/ decision transformer
                dec_input = {'rtg': torch.FloatTensor(rtg_list[-t:]),
                             'obs': torch.cat(obs_list[-t:], 1),
                             'act': torch.LongTensor(act_list[-t:])}
                import pdb
                pdb.set_trace()
                
                
                # eps-greedy
                with torch.no_grad():
                    logits, _ = self.model(obs)
                argmax_action = torch.argmax(logits, -1)[0].item()
                eps = self.cfg.eval_eps
                prob = random.random()
                
                if prob < eps:
                    action = random.randint(0, self.cfg.action_size-1)
                else:
                    action = argmax_action  

                # step
                next_obs, reward, done, info = self.env.step(action)

                # logger
                self.agent_logger.step(obs, reward, done, info)

                # move on
                if info.traj_done:
                    break
                else:
                    obs = next_obs
        
        log_data = self.agent_logger.fetch_log()
        
        return log_data
