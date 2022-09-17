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
    dataloader = loader(**cfg).get_dataloader()
    
    return dataloader