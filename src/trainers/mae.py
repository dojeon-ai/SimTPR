from .base import BaseTrainer
from src.common.vit_utils import get_random_1d_mask, get_random_3d_mask
from src.common.vis_utils import rollout_attn_maps
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import copy


class MAETrainer(BaseTrainer):
    name = 'mae'
    def __init__(self,
                 cfg,
                 device,
                 dataloader,
                 logger, 
                 aug_func,
                 model):
        
        super().__init__(cfg, device, dataloader, logger, aug_func, model)  
        self.cfg.patch_size = self.model.backbone.patch_size
        self.cfg.num_patches = self.model.backbone.num_patches
 
    def _get_patch_mask(self):
        # indiv-frame -> spatio-temporal MAE
        if self.cfg.process_type == 'indiv_frame':
            video_shape = (self.cfg.batch_size, self.cfg.t_step, self.cfg.num_patches) 
        
        # stack-frame -> image MAE
        elif self.cfg.process_type == 'stack_frame':
            video_shape = (self.cfg.batch_size, 1, self.cfg.num_patches) 
            
        patch_ids_keep, patch_mask, patch_ids_restore = get_random_3d_mask(video_shape, 
                                                                           self.cfg.patch_mask_ratio, 
                                                                           self.cfg.patch_mask_type)
        # to device
        patch_ids_keep = patch_ids_keep.to(self.device)
        patch_mask = patch_mask.to(self.device)
        patch_ids_restore = patch_ids_restore.to(self.device)
        
        patch_mask_dict = {
            'patch_mask_type': self.cfg.patch_mask_type,
            'patch_mask': patch_mask,
            'patch_ids_keep': patch_ids_keep,
            'patch_ids_restore': patch_ids_restore,
        }
        return patch_mask_dict

    def compute_loss(self, obs, act, rew, done):        
        ####################
        # augmentation
        x = obs / 255.0
        x = rearrange(x, 'n t c h w -> n (t c) h w')
        
        with torch.no_grad():
            aug_mask = torch.rand(x.shape[0], device=x.device) >= self.cfg.aug_prob
            aug_mask = rearrange(aug_mask, 'n -> n 1 1 1').float()
            aug_x = self.aug_func(x)
            x = aug_mask * x + (1-aug_mask) * aug_x
            
        x = rearrange(x, 'n (t c) h w -> n t c h w', t=self.cfg.t_step) 
        
        ####################
        # loss computation
        # get target
        if self.cfg.process_type == 'indiv_frame':
            patch = rearrange(x, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                              p1 = self.cfg.patch_size[0], p2 = self.cfg.patch_size[1])

        elif self.cfg.process_type == 'stack_frame':
            patch = rearrange(x, 'n t c (h p1) (w p2) -> n (h w) (t p1 p2 c)', 
                              p1 = self.cfg.patch_size[0], p2 = self.cfg.patch_size[1])

        # get patch mask
        patch_mask_dict = self._get_patch_mask()

        # predict masked patches
        x, _ = self.model.backbone(x, patch_mask_dict=patch_mask_dict)
        patch_pred, _ = self.model.head(x, patch_mask_dict=patch_mask_dict)

        # loss
        patch_loss = (patch_pred - patch) ** 2
        patch_loss = patch_loss.mean(dim=-1)

        # average over the removed patches
        patch_mask = patch_mask_dict['patch_mask']
        loss = (patch_loss * patch_mask).sum() / patch_mask.sum()

        # logs
        log_data = {}
        log_data['loss'] = loss.item()

        return loss, log_data
  
    def evaluate(self, obs, act, rew, done):
        obs = obs / 255.0
        with torch.no_grad():
            patch_mask_dict = self._get_patch_mask()  
            x, _ = self.model.backbone(obs, patch_mask_dict=patch_mask_dict)
            patch_pred, _ = self.model.head(x, patch_mask_dict=patch_mask_dict)
            
        if self.cfg.process_type == 'indiv_frame':
            patch = rearrange(obs, 'n t c (h p1) (w p2) -> n (t h w) (p1 p2 c)', 
                              p1 = self.cfg.patch_size[0], p2 = self.cfg.patch_size[1])
            def depatchify(patch):
                video = rearrange(patch, 'n (t h w) (p1 p2 c) -> n t c (h p1) (w p2)', 
                                  t=self.cfg.t_step, 
                                  c = self.cfg.obs_shape[1],
                                  h=self.cfg.obs_shape[2]//self.cfg.patch_size[0], 
                                  w=self.cfg.obs_shape[3]//self.cfg.patch_size[1], 
                                  p1 = self.cfg.patch_size[0], 
                                  p2 = self.cfg.patch_size[1])
                return video
        
        elif self.cfg.process_type == 'stack_frame':
            patch = rearrange(obs, 'n t c (h p1) (w p2) -> n (h w) (t p1 p2 c)', 
                              p1 = self.cfg.patch_size[0], p2 = self.cfg.patch_size[1])
            def depatchify(patch):
                video = rearrange(patch, 'n (h w) (t p1 p2 c) -> n t c (h p1) (w p2)', 
                                  t=self.cfg.t_step, 
                                  c = self.cfg.obs_shape[1],
                                  h=self.cfg.obs_shape[2]//self.cfg.patch_size[0], 
                                  w=self.cfg.obs_shape[3]//self.cfg.patch_size[1], 
                                  p1 = self.cfg.patch_size[0], 
                                  p2 = self.cfg.patch_size[1])
                return video
        
        # normalize prediction to [0,1]
        patch_mask = patch_mask_dict['patch_mask']
        patch_pred = patch_pred * patch_mask.unsqueeze(-1) + patch * (1-patch_mask).unsqueeze(-1)
        patch_pred = patch_pred.double()
        patch_pred = torch.where(patch_pred <0.0, 0.0, patch_pred)        
        patch_pred = torch.where(patch_pred>=1.0, 1.0, patch_pred)
        
        # depatchify
        masked_frames = depatchify(patch * (1-patch_mask).unsqueeze(-1))[0]
        pred_frames = depatchify(patch_pred)[0]
        target_frames = depatchify(patch)[0]
        
        log_data = {'masked_frames': wandb.Image(masked_frames),
                    'pred_frames': wandb.Image(pred_frames),
                    'target_frames': wandb.Image(target_frames)}
        return log_data
        