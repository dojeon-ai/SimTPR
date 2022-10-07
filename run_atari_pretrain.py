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
    parser.add_argument('--config_name',  type=str,    default='mixed_curl_impala') 
    parser.add_argument('--debug',        type=str2bool,   default=True)
    parser.add_argument('--num_seeds',     type=int,   default=1)
    parser.add_argument('--num_devices',   type=int,   default=4)
    parser.add_argument('--num_exp_per_device',  type=int,  default=1)
    parser.add_argument('--overrides',    action='append',  default=[]) 
    
    args = vars(parser.parse_args())
    seeds = np.arange(args.pop('num_seeds'))
    _games = list(atari_human_scores.keys())
    num_devices = args.pop('num_devices')
    num_exp_per_device = args.pop('num_exp_per_device')
    pool_size = num_devices * num_exp_per_device 
    debug = args.pop('debug')
    
    # mode
    mode = args.pop('mode')
    if mode == 'test':
        games = ['assault', 'asterix', 'boxing', 'frostbite', 
                 'demon_attack', 'gopher', 'seaquest', 'krull']
        
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

    # run parallell experiments
    # maxtasksperchild=1 -> no.of workers = no.of experiements
    # maxtasksperchild=None -> no.of workers = pool size
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    # mp.set_start_method('spawn') 
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    results = []
    for exps in list(chunks(experiments, pool_size)):
        pool = Pool(pool_size, maxtasksperchild=1)
        results_per_pool = pool.map(run, exps, chunksize=1)
        pool.close()
        pool.join()
        results += results_per_pool

    # artifacts from each logger
    artifacts_dict = {}
    for logger in results:
        artifacts = logger.get_artifacts()
        for path, name in artifacts.items():
            artifacts_dict[path] = name

    ####################
    # wandb
    wandb.init(project='atari_pretrain',
               entity='draftrec',
               config=args,
               group=args['exp_name'],
               settings=wandb.Settings(start_method="thread"))  
    artifact = wandb.Artifact(name=args['exp_name'], type='model')

    # save models
    for path, name in artifacts_dict.items():
        artifact.add_file(path, name=name)
    
    wandb.run.finish_artifact(artifact)
