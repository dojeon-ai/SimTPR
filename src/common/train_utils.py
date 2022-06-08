import numpy as np
import torch
import torch.nn as nn


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class LinearScheduler(object):
    def __init__(self, initial_value, final_value, step_size):
        """
        Linear Interpolation between initial_value to the final_value
        [params] initial_value (float) initial output value
        [params] final_value (float) final output value
        [params] step_size (int) number of timesteps to lineary anneal initial value to the final value
        """
        self.initial_value = initial_value
        self.final_value   = final_value
        self.step_size = step_size
        
    def get_value(self, step):
        """
        Return the scheduled value
        """
        interval = (self.initial_value - self.final_value) / self.step_size
        # After the schedule_timesteps, final value is returned
        if self.final_value < self.initial_value:
            return max(self.initial_value - interval * step, self.final_value)
        else:
            return min(self.initial_value - interval * step, self.final_value)