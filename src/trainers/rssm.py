from .base import BaseTrainer
from src.common.losses import ConsistencyLoss
from src.common.train_utils import LinearScheduler
from src.common.vit_utils import get_random_3d_mask, get_3d_masked_input, restore_masked_input
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import wandb

class RSSMTrainer(BaseTrainer):
    name = 'rssm'
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 eval_act_loader,
                 eval_rew_loader,
                 logger, 
                 aug_func,
                 model):
        
        super().__init__(cfg, device, 
                         train_loader, eval_act_loader, eval_rew_loader,
                         logger, aug_func, model)  
        self.target_model = copy.deepcopy(self.model).to(self.device)        
        update_steps = len(self.train_loader) * self.cfg.num_epochs
        cfg.tau_scheduler.step_size = update_steps
        self.tau_scheduler = LinearScheduler(**cfg.tau_scheduler)
        
    def compute_loss(self, obs, act, rew, done):
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
        
        # strong augmentation to online: (masking)
        assert self.cfg.mask_type in {'none', 'pixel'}
        if self.cfg.mask_type == 'none':
            x_o = x
            x_t = x
        
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
            
            x_t = x
        
        #################
        # forward
        # online encoder
        y_o, _ = self.model.backbone(x_o)        
        e_o = self.model.head.decode(y_o, act) # t=0: identity, t=1...T: predicted
        z_o = self.model.head.project(e_o)
        p_o = self.model.head.predict(z_o)
        z1_o, z2_o = z_o.chunk(2)
        p1_o, p2_o = p_o.chunk(2)
        
        # target encoder
        with torch.no_grad():
            y_t, _ = self.target_model.backbone(x_t)
            z_t = self.target_model.head.project(y_t)
            z1_t, z2_t = z_t.chunk(2)
        
        # action prediction
        act1_pred = self.model.head.act_predict(z1_o[:, :-1, :], z2_t[:, 1:, :])
        act2_pred = self.model.head.act_predict(z2_o[:, :-1, :], z1_t[:, 1:, :])

        #################
        # loss
        # self-predictive loss
        sp_loss_fn = ConsistencyLoss()
        p1_o, p2_o = rearrange(p1_o, 'n t d -> (n t) d'), rearrange(p2_o, 'n t d -> (n t) d')
        z1_t, z2_t = rearrange(z1_t, 'n t d -> (n t) d'), rearrange(z2_t, 'n t d -> (n t) d')

        sp_loss = 0.5 * (sp_loss_fn(p1_o, z2_t) + sp_loss_fn(p2_o, z1_t))
        sp_loss = torch.mean(sp_loss)

        # idm loss
        idm_loss_fn = nn.CrossEntropyLoss()
        act1_pred = rearrange(act1_pred, 'n t d -> (n t) d')
        act2_pred = rearrange(act2_pred, 'n t d -> (n t) d')
        
        act = act[:, :-1]
        act = rearrange(act, 'n t -> (n t)')
        act1, act2 = act.chunk(2)
        
        idm_loss = 0.5 * (idm_loss_fn(act1_pred, act1) + idm_loss_fn(act2_pred, act2))

        loss = sp_loss + self.cfg.idm_lmbda * idm_loss
        
        ###############
        # logs
        # quantitative
        pos_idx = torch.eye(p1_o.shape[0], device=p1_o.device)
        sim = F.cosine_similarity(p1_o.unsqueeze(1), z2_t.unsqueeze(0), dim=-1)
        pos_sim = (torch.sum(sim * pos_idx) / torch.sum(pos_idx))
        neg_sim = (torch.sum(sim * (1-pos_idx)) / torch.sum(1-pos_idx))
        pos_neg_diff = pos_sim - neg_sim
        
        # qualitative
        # (n, t, f, c, h, w) -> (t, c, h, w)
        masked_frames = x_o[0, :, 0, :, :, :]
        target_frames = x_t[0, :, 0, :, :, :]
        
        log_data = {'loss': loss.item(),
                    'sp_loss': sp_loss.item(),
                    'idm_loss': idm_loss.item(),
                    'pos_sim': pos_sim.item(),
                    'neg_sim': neg_sim.item(),
                    'pos_neg_diff': pos_neg_diff.item(),
                    'masked_frames': wandb.Image(masked_frames),
                    'target_frames': wandb.Image(target_frames)}
        
        return loss, log_data

    def update(self, obs, act, rew, done):
        tau = self.tau_scheduler.get_value()
        for online, target in zip(self.model.parameters(), self.target_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
