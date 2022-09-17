import ipdb
from collections import deque
from .base import BaseBuffer
import numpy as np
import torch
from src.common.train_utils import set_global_seeds


class DDPGBuffer(BaseBuffer):
    name = 'ddpg_buffer'
    def __init__(self, buffer_size, n_step, gamma, frame_stack,
                 data_specs=None, device='cpu'):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.n_step = n_step
        self.gamma = gamma
        self.full = False
        # fixed since we can only sample transitions that occur n_step earlier
        # than the end of each episode or the last recorded observation
        self.gamma_vec = np.power(gamma, np.arange(n_step)).astype('float32')
        self.next_dis = gamma**n_step
        self.device = device
        self.first = True

    def _initial_setup(self, obs, action, reward, done):
        self.index = 0
        self.obs_shape = list(obs.shape)
        self.ims_channels = 3
        self.act_shape = action.shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[-2:]], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.done = np.zeros([self.buffer_size], dtype=np.bool_)
        # which timesteps can be validly sampled 
        # 0,1,2->o_1, 3->o_2, ..., 501->o_500, 502,503,504->o_1, 505->o_2, ...
        # 0,1,2, 499,500,501(last3개),502,503(first두개(o_1을 복사한거)), 1001,1002,1003,1004,1005, ...
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)
        # 데이터셋 저장을 위한 np.array (last3개는 저장, 0,1은 저장 X)
        self.save_valid = np.ones([self.buffer_size], dtype=np.bool_)

    def add_data_point(self, obs, action, reward, done, first):      
        latest_obs = obs[-1]

        # store
        if self.first:
            # index=0 -> 
            # index 0,1,2 with first transition
            # self.index = 3
            end_index = self.index + self.frame_stack 
            end_invalid = end_index + self.frame_stack + 1 #TODO: 이거 같은데,  +1인가?

            # TODO: 밑에 sample에서 % buffer_size를 해주는데, 요 과정이 first에만 왜 필요한거지?
            # TODO: end_invalid > self.buffer_size에도, 사실은 self.save_valid = False 만들어주는 부분이 있어야함
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs

                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False

            else:
                self.valid[self.index:end_invalid] = False
                np.copyto(self.obs[self.index:end_index], latest_obs)
                np.copyto(self.act[self.index:end_index], action)
                np.copyto(self.rew[self.index:end_index], reward)
                np.copyto(self.dis[self.index:end_index], 1)
                self.save_valid[self.index:end_index-1] = False
                
            self.index = end_index
            self.traj_index = 1
            self.first = False

        else:
            np.copyto(self.obs[self.index], latest_obs)
            np.copyto(self.act[self.index], action)
            self.rew[self.index] = reward
            self.dis[self.index] = 1
            
            # idx = self.index + self.frame_stack -1가 sample 될 경우 
            # (e.g., 9(=sampled_idx)=7(=self.index)+3(=self.frame_stack)-1)
            # obs -> [obs[7], obs[8], obs[9]]가 sample 될건데
            # 여기서 obs[9]를 내가 수정해버렸기 때문에 (override)
            # valid[9]은 False가 된다.
            self.valid[(self.index + self.frame_stack -1) % self.buffer_size] = False  

            # idx = self.index - self.n_step +1이 sample 될 경우
            # (e.g., 4(sampled_idx) = 7(=self.index)-3(=self.n_step))
            # next_obs -> [obs[5], obs[6], obs[7]]가 sample 될건데
            # 내가 방금 obs[7]를 제대로 된 값으로 덮어씌웠기 때문에
            # valid[4]는 True가 된다.
            if self.traj_index >= self.n_step:  
                self.valid[(self.index - self.n_step) % self.buffer_size] = True

            # FIXME: self_valid가 False가 된 부분도, buffer를 한바퀴 돌면 다시 True가 될 수 있다
            self.save_valid[self.index] = 1
            
            self.index += 1               
            self.traj_index += 1

            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

        if done:
            self.done[self.index-1] = True
            self.first = True


    def store(self, obs, action, reward, done, first=False):
        if self.index == -1:
            self._initial_setup(obs, action, reward, done)
        self.add_data_point(obs, action, reward, done, first)

    def sample(self, batch_size):
        indices = np.random.choice(self.valid.nonzero()[0], size=batch_size)
        return self.gather_n_step_indices(indices)

    def gather_n_step_indices(self, indices):
        n_samples = indices.shape[0]

        # all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.n_step)
        #                           for i in range(n_samples)], axis=0) % self.buffer_size
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack+1, indices[i] + self.n_step+1)
                                  for i in range(n_samples)], axis=0) % self.buffer_size


        # gather_ranges = all_gather_ranges[:, self.frame_stack:] # bs x n_step
        gather_ranges = all_gather_ranges[:, self.frame_stack-1:-1] # bs x n_step
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]
        


        all_rewards = self.rew[gather_ranges]

        # Could implement reward computation as a matmul in pytorch for
        # marginal additional speed improvement
        rew = np.sum(all_rewards * self.gamma_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        ret = (obs, act, rew, dis, nobs)
        return ret

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def encode_obs(self, obs, prediction=False, ddpg=True):
        obs = np.array(obs).astype(np.float32)

        if ddpg:
            if prediction:  # FIXME:
                obs = obs / 255.0 - 0.5
        else:
            obs = obs / 255.0    

        if prediction:
            obs = np.expand_dims(obs, 0)

        N, S, C, W, H = obs.shape
        obs = obs.reshape(N, S*C, H, W)
        obs = torch.FloatTensor(obs).to(self.device)

        return obs

