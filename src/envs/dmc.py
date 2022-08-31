# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# from collections import deque
# from typing import Any, NamedTuple


import ipdb
from collections import deque
import gym
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels

from .base import *

import dmc2gym


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        # self.observation_space = gym.spaces.Box(
        #     low=0,
        #     high=1,
        #     shape=((shp[0] * k,) + shp[1:]),
        #     dtype=env.observation_space.dtype)
        self.observation_space = gym.spaces.Box(  # 추가
            low=0,
            high=1,
            shape=(k, *shp),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        obs = np.expand_dims(obs, axis=0)  # 추가
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.expand_dims(obs, axis=0)  # 추가
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def make_dmc_env(game, frame_stack, action_repeat, seed, num_timesteps, stddev_schedule):
    domain, task = game.split('_', 1)
    camera_id = 2 if domain == 'quadruped' else 0
    env = dmc2gym.make(domain_name=domain,
                       task_name=task,
                       seed=seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=84,
                       width=84,
                       frame_skip=action_repeat,
                       camera_id=camera_id)

    env = FrameStack(env, k=frame_stack)
    env.seed(seed)
    return env
