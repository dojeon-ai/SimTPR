import subprocess
import argparse
import json
import copy
import wandb
import itertools
import multiprocessing as mp
from multiprocessing import Pool
from src.common.class_utils import *
from src.envs.atari import *
from run_pretrain import run
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--group_name',type=str,   default='test')
    parser.add_argument('--exp_name',     type=str,    default='test')
    parser.add_argument('--mode',         type=str,    choices=['test','full'])
    parser.add_argument('--config_dir',   type=str,    default='atari/pretrain')
    parser.add_argument('--config_name',  type=str,    default='mixed_hicat_impala') 
    parser.add_argument('--debug',        type=str2bool,   default=True)
    parser.add_argument('--num_seeds',     type=int,   default=1)
    parser.add_argument('--num_devices',   type=int,   default=1)
    parser.add_argument('--num_exp_per_device',  type=int,  default=1)
    parser.add_argument('--overrides',    action='append',  default=[]) 
    
    args = vars(parser.parse_args())
    seeds = np.arange(args.pop('num_seeds'))
    games = list(atari_human_scores.keys())[:8]
    num_devices = args.pop('num_devices')
    num_exp_per_device = args.pop('num_exp_per_device')
    pool_size = num_devices * num_exp_per_device 
    debug = args.pop('debug')
    
    # mode
    mode = args.pop('mode')
        
    # create configurations for child run
    experiments = []
    device_id = 0
    for seed, game in itertools.product(*[seeds, games]):
        exp = copy.deepcopy(args)
        group_name = exp.pop('group_name')
        exp_name = exp.pop('exp_name')
        device_id = int(device_id % num_devices)
        camel_game = ''.join(word.title() for word in str(game).split('_'))
        
        exp['overrides'].append('group_name=' + group_name)
        exp['overrides'].append('exp_name=' + exp_name)
        exp['overrides'].append('seed=' + str(seed))
        exp['overrides'].append('dataloader.game=' + str(camel_game))
        exp['overrides'].append('device=' + 'cuda:' + str(device_id))
        exp['overrides'].append('debug=' + str(debug))

        experiments.append(exp)
        device_id += 1
        print(exp)
        
    for exp in experiments:
        result = run(exp)
        

    # run parallell experiments
    # maxtasksperchild=1 -> no.of workers = no.of experiements
    # maxtasksperchild=None -> no.of workers = pool size
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    # mp.set_start_method('spawn') 
    
    import pdb
    pdb.set_trace()
    
    
    