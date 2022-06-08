from .backbones import *  
from .headers import * 
from .policies import *
from .base import Model
from omegaconf import OmegaConf
from src.common.utils import all_subclasses


BACKBONES = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseBackbone)}

HEADERS = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseHeader)}

POLICIES = {subclass.get_name():subclass
            for subclass in all_subclasses(BasePolicy)}


def build_model(cfg):
    cfg = OmegaConf.to_container(cfg)
    backbone_cfg = cfg['backbone']
    header_cfg = cfg['header']
    policy_cfg = cfg['policy']
    
    backbone_type = backbone_cfg.pop('type')
    header_type = header_cfg.pop('type')
    policy_type = policy_cfg.pop('type')

    backbone = BACKBONES[backbone_type]
    backbone = backbone(**backbone_cfg)
    if header_type != str(None):
        header = HEADERS[header_type]
        header = header(**header_cfg)
    else:
        header = None
    policy = POLICIES[policy_type]
    policy = policy(**policy_cfg)

    model = Model(backbone = backbone,
                  header = header,
                  policy = policy)
    
    return model