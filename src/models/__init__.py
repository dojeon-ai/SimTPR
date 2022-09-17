from .backbones import *  
from .heads import * 
from .policies import *
from .base import TrainerModel, AgentModel
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses
import torch


BACKBONES = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseBackbone)}

HEADS = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseHead)}

POLICIES = {subclass.get_name():subclass
            for subclass in all_subclasses(BasePolicy)}


def build_trainer_model(cfg):
    cfg = OmegaConf.to_container(cfg)
    backbone_cfg = cfg['backbone']
    head_cfg = cfg['head']
    
    backbone_type = backbone_cfg.pop('type')
    head_type = head_cfg.pop('type')

    # backbone
    backbone = BACKBONES[backbone_type]
    backbone = backbone(**backbone_cfg)
    
    # get output dim of backbone
    fake_obs = torch.zeros((1, *backbone_cfg['obs_shape']))
    out, _ = backbone(fake_obs)
    output_dim = out.shape[-1]
    
    # head
    head_cfg['in_features'] = output_dim
    head = HEADS[head_type]
    head = head(**head_cfg)

    model = TrainerModel(backbone=backbone, head=head)
    
    return model


def build_agent_model(cfg):
    cfg = OmegaConf.to_container(cfg)
    backbone_cfg = cfg['backbone']
    neck_cfg = cfg['neck']
    policy_cfg = cfg['policy']
    
    backbone_type = backbone_cfg.pop('type')
    neck_type = neck_cfg.pop('type')
    policy_type = policy_cfg.pop('type')

    # backbone
    backbone = BACKBONES[backbone_type]
    backbone = backbone(**backbone_cfg)
    
    # get output dim of backbone
    fake_obs = torch.zeros((1, *backbone_cfg['obs_shape']))
    output_dim = backbone(fake_obs).shape[-1]

    # head
    neck_cfg['in_features'] = output_dim
    neck = NECKS[neck_type]
    neck = neck(**neck_cfg)
    
    # policy
    policy = POLICIES[policy_type]
    policy = policy(**policy_cfg)

    model = AgentModel(backbone=backbone, neck=neck, policy=policy)
    
    return model

