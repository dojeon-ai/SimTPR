from .backbones import *  
from .heads import * 
from .policies import *
from .base import Model
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
    if head_type != str(None):
        head = HEADS[head_type]
        head = head(**head_cfg)
    else:
        head = None
    policy = POLICIES[policy_type]
    policy = policy(**policy_cfg)

    model = Model(backbone = backbone,
                  head = head,
                  policy = policy)
    
    return model