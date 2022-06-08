from collections import deque
from .base import BaseBuffer
import numpy as np
import torch

class DQNBuffer(BaseBuffer):
    name = 'dqn_buffer'
    """ 
    Replay buffer stores the trajectories and encode those to feed it to the neural network
    Several memory optimization methods are listed below
    1. Since the frame is managed with integer datatype, scaling the frame into [0,1] is only done before passed to hte model. 
    Storing the observation is seperated from storing the other information in order to efficiently store the next observation.
    """ 
    def __init__(self, size, n_step, gamma, device):
        """
        [params] size (int) size of the replay buffer
        [params] obs_shape (list) (num_imb_obs, num_channels, width, height)
        """
        # Initialize
        self.size = size
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.num_in_buffer = 0
        self.n_step_transitions = deque(maxlen=self.n_step)
        self.transitions = deque(maxlen=self.size)
        
    def _get_n_step_info(self):
        transitions = list(self.n_step_transitions)
        obs, action, G, done, next_obs = transitions[-1]
        for _, _, _reward, _done, _next_obs in reversed(transitions[:-1]):
            G = _reward + self.gamma * G * (1-_done)    
            if _done:
                done, next_obs = _done, _next_obs
        
        return (obs, action, G, done, next_obs)
        
    def store(self, obs, action, reward, done, next_obs):
        self.n_step_transitions.append((obs, action, reward, done, next_obs))
        if len(self.n_step_transitions) < self.n_step:
            return
        transition = self._get_n_step_info()
        self.transitions.append(transition)
        self.num_in_buffer = min(self.num_in_buffer+1, self.size)

    # sample transitions for model training
    def sample(self, batch_size):
        if self.num_in_buffer < batch_size:
            assert('Replay buffer does not have enough transitions to sample')
        idxs = np.random.choice(np.arange(self.num_in_buffer-1), batch_size, replace = False)
        transitions = [self.transitions[idx] for idx in idxs]
        obs_batch, act_batch, rew_batch, done_batch, next_obs_batch = zip(*transitions)

        obs_batch = self.encode_obs(obs_batch)  
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        next_obs_batch = self.encode_obs(next_obs_batch)
        # weights are used to keep compatibility with prioritzed buffer
        weights = torch.ones(batch_size).to(self.device)

        return idxs, obs_batch, act_batch, rew_batch, done_batch, next_obs_batch, weights
    
    def encode_obs(self, obs, prediction=False):
        obs = np.array(obs).astype(np.float32)
        obs = obs / 255.0

        if prediction:
            obs = np.expand_dims(obs, 0)

        N, S, C, W, H = obs.shape
        obs = obs.reshape(N, S*C, H, W)
        obs = torch.FloatTensor(obs).to(self.device)

        return obs
