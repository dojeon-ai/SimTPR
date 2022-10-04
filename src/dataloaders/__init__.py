from .base import BaseLoader
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseLoader)

LOADERS = {subclass.get_name():subclass
          for subclass in all_subclasses(BaseLoader)}

def build_dataloader(cfg):
    cfg = OmegaConf.to_container(cfg)
    loader_type = cfg.pop('type')
    loader = LOADERS[loader_type]
    
    train_cfg = cfg.pop('train')
    eval_act_cfg = cfg.pop('act')
    eval_rew_cfg = cfg.pop('rew')
    
    train_cfg.update(cfg)
    eval_act_cfg.update(cfg)
    eval_rew_cfg.update(cfg)

    train_loader = loader(**train_cfg).get_dataloader()
    eval_act_loader = loader(**eval_act_cfg).get_dataloader()
    eval_rew_loader = loader(**eval_rew_cfg).get_dataloader()
    
    return train_loader, eval_act_loader, eval_rew_loader