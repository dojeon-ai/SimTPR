import gzip
import re
import os
from pathlib import Path
from typing import List, Tuple
from itertools import zip_longest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .base import BaseLoader
from src.envs.atari import AtariEnv
from src.common.data_utils import *


class DQNReplayDataset(Dataset):
    def __init__(self, 
                 data_path: Path,
                 tmp_data_path: Path,
                 game: str,
                 checkpoint: int,
                 frames: int,
                 k_step: int,
                 max_size: int,
                 full_action_set: bool,
                 dataset_on_gpu: bool,
                 dataset_on_disk: bool) -> None:

        data = []
        self.dataset_on_disk = dataset_on_disk
        assert not (dataset_on_disk and dataset_on_gpu)
        filetypes = ['observation', 'action', 'reward', 'terminal']
        for i, filetype in enumerate(filetypes):
            filename = Path(data_path + '/' + f'{game}/{filetype}_{checkpoint}.gz')
            print(f'Loading {filename}')

            # There's no point in putting rewards, actions or terminals on disk.
            # They're tiny and it'll just cause more I/O.
            on_disk = dataset_on_disk and filetype == "observation"
            g = gzip.GzipFile(filename=filename)
            data__ = np.load(g)
            #if i == 0:
            #    self.has_parallel_envs = len(data__.shape) > 1
            #    if self.has_parallel_envs:
            #        self.n_envs = data__.shape[1]
            #    else:
            #        self.n_envs = 1
            #if not self.has_parallel_envs:
            # data__ = np.expand_dims(data__, 1)

            # number of interactions for each checkpoint
            data___ = np.copy(data__[:max_size])
            print(f'Using {data___.size * data___.itemsize} bytes')
            if not on_disk:
                del data__
                data_ = torch.from_numpy(data___)
            else:
                new_filename = tmp_data_path + '/' + game
                new_filename = os.path.join(new_filename, Path(os.path.basename(filename)[:-3]+".npy"))
                print("Stored on disk at {}".format(new_filename))
                np.save(new_filename, data___,)
                del data___
                del data__
                data_ = np.load(new_filename, mmap_mode="r+")

            if (filetype == 'action') and full_action_set:
                action_mapping = dict(zip(data_.unique().numpy(),
                                        AtariEnv(re.sub(r'(?<!^)(?=[A-Z])', '_', game).lower()).ale.getMinimalActionSet()))
                data_.apply_(lambda x: action_mapping[x])
            if dataset_on_gpu:
                print("Stored on GPU")
                data_ = data_.cuda(non_blocking=True)
                del data___
            data.append(data_)
            setattr(self, filetype, data_)

        self.game = game
        self.f = frames
        self.k = k_step
        self.size = min(self.action.shape[0], max_size)
        self.effective_size = (self.size - self.f - self.k + 1)

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # batch_ind = index // self.effective_size
        time_ind = index % self.effective_size
        sl = slice(time_ind, time_ind+self.f+self.k)
        if self.dataset_on_disk:
            obs = torch.from_numpy(self.observation[sl])
        else:
            obs = (self.observation[sl])

        return tuple([obs,
                    self.action[sl],
                    self.reward[sl],
                    self.terminal[sl]])


class MultiDQNReplayDataset(Dataset):
    def __init__(self, 
                data_path: Path,
                tmp_data_path: Path, 
                game: str,
                checkpoints: List[int],
                frames: int,
                k_step: int,
                max_size: int,
                full_action_set: bool,
                dataset_on_gpu: bool,
                dataset_on_disk: bool) -> None:
        
        self.datasets =[DQNReplayDataset(data_path,
                        tmp_data_path,
                        game,
                        ckpt,
                        frames,
                        k_step,
                        max_size,
                        full_action_set,
                        dataset_on_gpu,
                        dataset_on_disk) for ckpt in checkpoints]

        self.num_blocks = len(self.datasets)
        self.block_len = len(self.datasets[0])

    def __len__(self) -> int:
        return len(self.datasets) * len(self.datasets[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ckpt_index = index % len(self.datasets)
        index = index // len(self.datasets)
        return self.datasets[ckpt_index][index]


class ATARILoader(BaseLoader):
    name = 'atari'
    def __init__(self,
                 data_path: Path,
                 tmp_data_path: Path,
                 game: str,
                 checkpoints: List[int],
                 frames: int,
                 k_step: int, # length of the future trajectory to predict
                 max_size: int,
                 dataset_on_gpu: bool,
                 dataset_on_disk: bool,
                 batch_size: int,
                 full_action_set: bool,
                 num_workers: int,
                 pin_memory: bool,
                 prefetch_factor: int,
                 group_read_factor: int=0,
                 shuffle_checkpoints: bool=False):
        
        super().__init__()
        self.data_path = data_path
        self.tmp_data_path = tmp_data_path
        self.game = game
        self.checkpoints = checkpoints
        self.frames = frames
        self.k_step = k_step
        self.max_size = max_size
        self.dataset_on_gpu = dataset_on_gpu
        self.dataset_on_disk = dataset_on_disk
        self.batch_size = batch_size
        self.full_action_set = full_action_set
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.group_read_factor = group_read_factor
        self.shuffle_checkpoints = shuffle_checkpoints

    def get_dataloader(self):
        def collate(batch):
            frames = self.frames
            observation, action, reward, done = torch.utils.data.dataloader.default_collate(batch)
            # tbcfhw: standard format for ATC
            observation = torch.einsum('bthw->tbhw', observation).unsqueeze(2).repeat(1, 1, frames, 1, 1)
            for i in range(1, frames):
                observation[:, :, i] = observation[:, :, i].roll(-i, 0)
            observation = observation[:-frames] # do not use the last frame (just to make t=k)
            observation = observation.unsqueeze(3)
            action = torch.einsum('bt->tb', action)[frames-1:-1].long()
            reward = torch.einsum('bt->tb', reward)[frames-1:-1]
            reward = torch.nan_to_num(reward).sign()  # Apparently possible, somehow.
            done = torch.einsum('bt->tb', done)[frames-1:-1].bool()

            return sanitize_batch(OfflineSamples(observation, action, reward, done))

        dataset = MultiDQNReplayDataset(self.data_path, 
                                        self.tmp_data_path, 
                                        self.game, 
                                        self.checkpoints, 
                                        self.frames, 
                                        self.k_step, 
                                        self.max_size,
                                        self.full_action_set, 
                                        self.dataset_on_gpu, 
                                        self.dataset_on_disk)

        if self.shuffle_checkpoints:
            data = get_from_dataloaders(dataset.datasets)
            shuffled_data = shuffle_batch_dim(*data)
            assign_to_dataloaders(dataset.datasets, *shuffled_data)

        if self.group_read_factor != 0:
            sampler = CacheEfficientSampler(dataset.num_blocks, dataset.block_len, self.group_read_factor)
            dataloader = DataLoader(dataset, 
                                    batch_size=self.batch_size,
                                    sampler=sampler,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    collate_fn=collate,
                                    drop_last=True,
                                    prefetch_factor=self.prefetch_factor)
        else:
            dataloader = DataLoader(dataset, 
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    collate_fn=collate,
                                    drop_last=True,
                                    prefetch_factor=self.prefetch_factor)

        return dataloader
