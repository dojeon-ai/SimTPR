import subprocess
import argparse
import json
import copy
import wandb
import itertools
import multiprocessing as mp
from multiprocessing import Pool
from src.envs.atari import *
from run_pretrain import run
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--exp_name',     type=str,    default='simclr',  help='name for the parallel runs')
    parser.add_argument('--config_dir',   type=str,    default='atari')
    parser.add_argument('--config_name',  type=str,    default='mixed_simclr_nature') 
    parser.add_argument('--num_seeds',     type=int,   default=1)
    parser.add_argument('--num_devices',   type=int,   default=3)
    parser.add_argument('--num_exp_per_device',  type=int,  default=1)
    parser.add_argument('--overrides',    action='append',  default=[]) #'agent.num_timesteps=2500', 'agent.eval_every=2500']) 
    
    args = vars(parser.parse_args())
    seeds = np.arange(args.pop('num_seeds'))
    _games = ['alien', 'asterix', 'ms_pacman', 'pong', 'qbert', 'seaquest']
    num_devices = args.pop('num_devices')
    num_exp_per_device = args.pop('num_exp_per_device')
    pool_size = num_devices * num_exp_per_device 

    # snake case to camel case
    games = []
    for _game in _games:
        game = ''.join(word.title() for word in _game.split('_'))
        games.append(game)

    # create configurations for child run
    experiments = []
    device_id = 0
    for seed, game in itertools.product(*[seeds, games]):
        exp = copy.deepcopy(args)
        exp_name = exp.pop('exp_name')
        device_id = int(device_id % num_devices)
        exp['overrides'].append('exp_name=' + exp_name)
        exp['overrides'].append('seed=' + str(seed))
        exp['overrides'].append('dataloader.game=' + str(game))
        exp['overrides'].append('device=' + 'cuda:' + str(device_id))

        experiments.append(exp)
        device_id += 1
        print(exp)

    # run parallell experiments
    # maxtasksperchild=1 -> no.of workers = no.of experiements
    # maxtasksperchild=None -> no.of workers = pool size
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    # mp.set_start_method('spawn') 
    pool = Pool(pool_size, maxtasksperchild=1)
    results = pool.map(run, experiments, chunksize=1)
    pool.close()

    # artifacts from each logger
    artifacts_dict = {}
    for logger in results:
        artifacts = logger.get_artifacts()
        for path, name in artifacts.items():
            artifacts_dict[path] = name

    ####################
    # wandb
    wandb.init(project='atari100k', 
               config=args,
               group=args['exp_name'],
               settings=wandb.Settings(start_method="thread"))  
    artifact = wandb.Artifact(name=args['exp_name'], type='model')

    # save models
    for path, name in artifacts_dict.items():
        artifact.add_file(path, name=name)
    
    wandb.run.finish_artifact(artifact)
