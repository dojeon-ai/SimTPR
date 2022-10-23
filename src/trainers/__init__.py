from .base import BaseTrainer
from dotmap import DotMap
from omegaconf import OmegaConf
from src.common.augmentation import Augmentation
from src.common.augmentation import RandomShiftsAug
from src.common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseTrainer)

TRAINERS = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseTrainer)}


def build_trainer(cfg,
                  device,
                  train_loader,
                  eval_act_loader,
                  eval_rew_loader,
                  env,
                  logger,
                  agent_logger,
                  model):
    
    cfg = DotMap(OmegaConf.to_container(cfg))

    # augemntation
    if len(cfg.aug_types) == 0:
        cfg.aug_types = []

    if 'dmc_random_crop' in cfg.aug_types:
        aug_func = RandomShiftsAug(pad=4)
    else:
        aug_func = Augmentation(obs_shape=cfg.obs_shape, 
                                aug_types=cfg.aug_types)

    # trainer
    trainer_type = cfg.pop('type')
    trainer = TRAINERS[trainer_type]
    return trainer(cfg=cfg,
                   device=device,
                   train_loader=train_loader,
                   eval_act_loader=eval_act_loader,
                   eval_rew_loader=eval_rew_loader,
                   env=env,
                   logger=logger,
                   agent_logger=agent_logger,
                   aug_func=aug_func,
                   model=model)
