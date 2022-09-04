from .backbones import *  
from .heads import * 
from .policies import *
from .base import DDPGModel, Model
from omegaconf import OmegaConf
from src.common.utils import all_subclasses


BACKBONES = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseBackbone)}

HEADS = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseHead)}

POLICIES = {subclass.get_name():subclass
            for subclass in all_subclasses(BasePolicy)}


def build_model(cfg):
    cfg = OmegaConf.to_container(cfg)
    backbone_cfg = cfg['backbone']
    head_cfg = cfg['head']
    policy_cfg = cfg['policy']
    
    backbone_type = backbone_cfg.pop('type')
    head_type = head_cfg.pop('type')
    policy_type = policy_cfg.pop('type')

    backbone = BACKBONES[backbone_type]
    backbone = backbone(**backbone_cfg)
    
    # TODO: erase drqv2 specific condition
    if head_type != str(None):
        head = HEADS[head_type]
        if backbone_type == 'drqv2_encoder':
            head_cfg['in_features'] = backbone.repr_features
        head = head(**head_cfg)
    else:
        head = None
    if policy_type != str(None):
        policy = POLICIES[policy_type]
        if backbone_type == 'drqv2_encoder':  # DrQ-v2 implementation
            policy_cfg['repr_features'] = backbone.repr_features 
        policy = policy(**policy_cfg)
    else:
        policy = None

    if backbone_type == 'drqv2_encoder':
        model = DDPGModel
    else:
        model = Model

    model = model(backbone=backbone, head=head, policy=policy)
    return model

