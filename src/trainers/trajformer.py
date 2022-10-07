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

class TrajFormerTrainer(BaseTrainer):
    name = 'trajformer'
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
        rew = torch.cat([rew, rew], axis=0)
        
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
        
        # decode
        dataset_type = self.cfg.dataset_type 
        if dataset_type == 'video':
            dec_input = {'obs': y_o}
            
        elif dataset_type == 'demonstration':
            dec_input = {'obs': y_o,
                         'act': act}
            
        elif dataset_type == 'trajectory':
            dec_input = {'obs': y_o,
                         'act': act,
                         'rew': rew}
    
        dec_output = self.model.head.decode(dec_input, dataset_type) 
        obs_o, act_o, rew_o = dec_output['obs'], dec_output['act'], dec_output['rew']        
        z_o = self.model.head.project(obs_o)
        p_o = self.model.head.predict(z_o)
        z1_o, z2_o = z_o.chunk(2)
        p1_o, p2_o = p_o.chunk(2)
        
        # target encoder
        with torch.no_grad():
            y_t, _ = self.target_model.backbone(x_t)
            z_t = self.target_model.head.project(y_t)
            z1_t, z2_t = z_t.chunk(2)

        #################
        # loss
        # observation loss (self-predictive)
        obs_loss_fn = ConsistencyLoss()
        p1_o, p2_o = rearrange(p1_o, 'n t d -> (n t) d'), rearrange(p2_o, 'n t d -> (n t) d')
        z1_t, z2_t = rearrange(z1_t, 'n t d -> (n t) d'), rearrange(z2_t, 'n t d -> (n t) d')

        obs_loss = 0.5 * (obs_loss_fn(p1_o, z2_t) + obs_loss_fn(p2_o, z1_t))
        obs_loss = torch.mean(obs_loss)

        # action loss
        if act_o is not None:
            act_loss_fn = nn.CrossEntropyLoss()
            act_o = rearrange(act_o, 'n t d -> (n t) d')
            act_t = rearrange(act, 'n t -> (n t)')
            act_loss = act_loss_fn(act_o, act_t)
            act_acc = torch.mean((torch.argmax(act_o, 1) == act_t).float())
        else:
            act_loss = torch.Tensor([0.0]).to(x.device)
            act_acc = torch.Tensor([0.0]).to(x.device)
            
        # idm loss
        idm_loss_fn = nn.CrossEntropyLoss()
        idm_o = self.model.head.idm_predictor(torch.cat((z_o[:, :-1, :], z_t[:, 1:, :]), dim=-1))
        idm_t = act[:, :-1]
        idm_o = rearrange(idm_o, 'n t d -> (n t) d')
        idm_t = rearrange(idm_t, 'n t -> (n t)')
        idm_loss = idm_loss_fn(idm_o, idm_t)
        idm_acc = torch.mean((torch.argmax(idm_o, 1) == idm_t).float())
        
        # reward loss
        if rew_o is not None:
            rew_loss_fn = nn.MSELoss()
            rew_o = rearrange(rew_o, 'n t 1 -> (n t 1)')
            rew = rearrange(rew, 'n t -> (n t)')
            rew_loss = rew_loss_fn(rew_o, rew)
        else:
            rew_loss = torch.Tensor([0.0]).to(x.device)

        loss = (self.cfg.obs_lmbda * obs_loss + self.cfg.act_lmbda * act_loss + 
                self.cfg.idm_lmbda * idm_loss + self.cfg.rew_lmbda * rew_loss)
        
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
                    'obs_loss': obs_loss.item(),
                    'act_loss': act_loss.item(),
                    'idm_loss': idm_loss.item(),
                    'rew_loss': rew_loss.item(),
                    'act_acc': act_acc.item(),
                    'idm_acc': idm_acc.item(),
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
