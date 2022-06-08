import argparse
import yaml
import os
from copy import deepcopy


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Config():
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv

    def parse(self):
        import pdb
        pdb.set_trace()
        

"""
class Parser:
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv

    def set_template(self, configs):
        template_name = configs['template']
        assert template_name is not None, 'template is not given'
        template = yaml.safe_load(open(os.path.join('templates', f'{template_name}.yaml')))
        # overwrite_with_non_nones
        for k, v in configs.items():
            if v is None:
                configs[k] = template[k]
        return configs

    def parse(self):
        # 1. initialize null arguments (parser should not contain default arguments)
        # 2. fill the arguments with cmd interface
        # 3. fill the arguments from templates
        priority: cmd > template
        configs = {}
        configs.update(self.parse_top())
        configs.update(self.parse_agent())
        configs = self.set_template(configs)
        return configs

    def parse_top(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--template', type=str)
        parser.add_argument('--project_name',  type=str, help='Wandb project name')
        parser.add_argument('--exp_name',      type=str, help='experiment name')
        parser.add_argument('--env_name',      type=str, help='environment ID')
        parser.add_argument('--obs_shape',     type=int, nargs='+', help='shape of the observation')
        parser.add_argument('--action_size',   type=int, nargs='+', help='size of the action')        
        parser.add_argument('--device',        type=str, required = False, help='whether to use gpu')
        parser.add_argument('--gpu_device',    type=int, required = False, help='visible device in CUDA')
        parser.add_argument('--seed',          type=int, help='random seed')
        parser.add_argument('--num_timesteps', type=int, help='overwrite the number of training timesteps')
        parser.add_argument('--num_eval_trajectories', type=str, help='number of trajectories to evaluate')
        parser.add_argument('--log_every',     type=int, help='frequecy of logging on wandb')        
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_agent(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        # model
        parser.add_argument('--agent_type',    type=str, choices=['dqn', ])
        parser.add_argument('--backbone_type', type=str, choices=['nature'], help='backbone architecture')
        parser.add_argument('--header_type',   type=str, choices=[''], help='header type')
        parser.add_argument('--policy_type',   type=str, choices=['dqn_head, rainbow_head'], help='policy type')
        parser.add_argument('--double',        type=str2bool, help='whether to use double-q')
        parser.add_argument('--n_step',        type=int,   help='n-step return to aggregate target')
        # exploration (if noisy layer is used, epsilon is not utilized)
        parser.add_argument('--noise_std',    type=float, help='standard deviation ')
        parser.add_argument('--eps_start',    type=float, help='initial value of epsilon')
        parser.add_argument('--eps_end',      type=float, help='final value of epsilon')
        parser.add_argument('--decay_step',   type=float, help='the number of time steps which eps linearly anneald to final value')
        parser.add_argument('--eval_eps',     type=float, help='fixed epsilon value at test time')
        # optimizer
        parser.add_argument('--optimizer',    type=str,   choices=['rmsprop', 'adam'])
        parser.add_argument('--lr',           type=float, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, help='l2 regularization')
        parser.add_argument('--beta1',        type=float, help='beta1 (adam), momentum (rmsprop)')
        parser.add_argument('--beta2',        type=float, help='beta2 (adam), alpha (rmsprop)')
        parser.add_argument('--eps',          type=float, help='epsilon')
        # update
        parser.add_argument('--gamma',        type=float, help='future rewards decay')
        parser.add_argument('--batch_size',   type=int,   help='mini-batch size to sample')
        parser.add_argument('--update_freq',        type=int, help='update frequency of the main network')
        parser.add_argument('--target_update_freq', type=int, help='update frequency of the target network')
        parser.add_argument('--buffer_size',        type=int, help='size of the replay buffer')
        parser.add_argument('--min_buffer_size',    type=int, help='minimum number of transitions in buffer to start update')
        parser.add_argument('--clip_grad_norm',     type=float)
        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)
"""