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


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S