from .base import BaseTrainer
from src.common.train_utils import LinearScheduler, CosineAnnealingWarmupRestarts
from src.common.vis_utils import rollout_attn_maps
from skimage.feature import hog
import skimage as sk
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
from fractions import Fraction
import random
import copy
import tqdm
import cv2


class ReconTrainer(BaseTrainer):
    name = 'recon'
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
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.num_epochs, cfg.scheduler)
        
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

    def _build_scheduler(self, optimizer, num_epochs, scheduler_cfg):
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             first_cycle_steps=num_epochs,
                                             **scheduler_cfg)
 
    def compute_loss(self, obs, act, done, flow, hog):  
        """
        [param] obs
        [param] act
        [param] done
        [param] flow
        [param] hog
        """
        n, t, s, c, h, w = obs.shape
        obs = obs / 255.0       
        
        # augmentation
        # perform identical augmentation over the temporal dimension
        obs = rearrange(obs, 'n t s c h w -> n (t s c) h w')
        with torch.no_grad():
            aug_mask = torch.rand(obs.shape[0], device=obs.device) >= self.cfg.aug_prob
            aug_mask = rearrange(aug_mask, 'n -> n 1 1 1').float()
            aug_obs = self.aug_func(obs)
            obs = aug_mask * obs + (1-aug_mask) * aug_obs
        
        # get target
        obs = rearrange(obs, 'n (t s c) h w -> (n t) (s c) h w', t=t, s=s, c=c) 
        flow = flow[:, :-1] 
        flow = rearrange(flow, 'n t c h w -> (n t) c h w')
        hog = rearrange(hog, 'n t d -> (n t) d')        
        
        # forward
        z = self.model.backbone(obs)
        z = rearrange(z, '(n t) d -> n t d', t=self.cfg.t_step)
        patch_pred, hog_pred, flow_pred = self.model.head(z)
        
        # compute loss
        patch_loss = (patch_pred - obs) ** 2
        hog_loss = (hog_pred - hog) ** 2
        flow_loss = (flow_pred - flow) **2
        
        # aggregate
        patch_loss = torch.mean(patch_loss)
        hog_loss = torch.mean(hog_loss)
        flow_loss = torch.mean(flow_loss)
        loss = self.cfg.patch_lmbda * patch_loss + self.cfg.hog_lmbda * hog_loss + self.cfg.flow_lmbda * flow_loss
        
        # log
        log_data = {}
        log_data['patch_loss'] = patch_loss.item()
        log_data['hog_loss'] = hog_loss.item()
        log_data['flow_loss'] = flow_loss.item()
        log_data['loss'] = loss.item()
        
        #patch_loss = patch_pred - x
        
        import pdb
        pdb.set_trace()
        
        """
        img = obs[0][0][0]
        img = rearrange(img, 'c h w -> h w c')
        img = img.cpu().numpy()
        
        for i in tqdm.tqdm(range(256*8)):
            hog_feat = sk.feature.hog(img, 
                                      orientations=9, 
                                      pixels_per_cell=(16, 16),
                                      cells_per_block=(3, 3), 
                                      visualize=False, 
                                      multichannel=True)

        """
        
        return loss, log_data
    
    def visualize(self, obs):
        obs = rearrange(obs, 'n t s c h w -> n t h w (s c)')
        obs = obs.cpu().numpy()        
        n, t, h, w, _ = obs.shape
        
        # iterate over timesteps
        frames = obs[0]
        flow_maps = np.zeros((t-1, h, w, 3))
        rgb_frames = np.zeros((t-1, h, w, 3))
            
        for t_step in range(t-1):
            _prev = frames[t_step]
            _next = frames[t_step+1]
            
            # get flow-map
            flow = cv2.calcOpticalFlowFarneback(prev=_prev,
                                                next=_next,
                                                flow=None,
                                                pyr_scale=0.5,
                                                levels=3, 
                                                winsize=6,
                                                iterations=20,
                                                poly_n=7,
                                                poly_sigma=1.5,
                                                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 
            
            # get hsv flow-map
            flow = flow
            mag, ang = cv2.cartToPolar(flow[:, :,0], flow[:, :,1])
            hsv = np.zeros((h, w, 3))
            hsv[:, :,0] = ang*180/np.pi/2
            hsv[:, :,1] = 255
            hsv[:, :,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)        
            flow_map = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            flow_maps[t_step] = flow_map

            rgb_frame = cv2.cvtColor(_prev,cv2.COLOR_GRAY2RGB)
            rgb_frames[t_step] = rgb_frame
            
        rgb_frames = torch.from_numpy(rgb_frames) / 255.0
        rgb_frames = rearrange(rgb_frames, 't h w c -> t c h w')
        frames = torch.from_numpy(frames) / 255.0
        frames = rearrange(frames, 't h w c -> t c h w')
        flow_maps = torch.from_numpy(flow_maps)
        flow_maps = rearrange(flow_maps, 't h w c -> t c h w')
        
        flow_video = (rgb_frames) + 0.002*(flow_maps)

        wandb.log({'video': wandb.Image(frames),
                   'flow_maps': wandb.Image(flow_maps),
                   'flow_video': wandb.Image(flow_video)
                  }, step=self.logger.step)
   
    def train(self):
        self.model.train()
        loss, t = 0, 0
        for e in range(1, self.cfg.num_epochs+1):
            for batch in tqdm.tqdm(self.dataloader):                         
                # forward
                obs = batch.observation.to(self.device)
                act = batch.action.to(self.device)
                done = batch.done.to(self.device)
                flow = batch.flow.to(self.device)
                hog = batch.hog.to(self.device)

                loss, log_data = self.compute_loss(obs, act, done, flow, hog)
                
                # backward
                """
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm.append(p.grad.detach().data.norm(2))
                grad_norm = torch.stack(grad_norm)
                log_data['min_grad_norm'] = torch.min(grad_norm).item()
                log_data['mean_grad_norm'] = torch.mean(grad_norm).item()
                log_data['max_grad_norm'] = torch.max(grad_norm).item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()
                """
                # evaluation
                if t % self.cfg.eval_every == 0:
                    self.visualize(obs)
                    
                # log
                log_data = {}
                self.logger.update_log(**log_data)
                if t % self.cfg.log_every == 0:
                    self.logger.write_log()

                # proceed
                t += 1
            if e % self.cfg.save_every == 0:
                self.logger.save_state_dict(model=self.model, epoch=e)

            self.lr_scheduler.step()
