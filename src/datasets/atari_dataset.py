import gzip
import re
import os
from pathlib import Path
from typing import List, Tuple
from itertools import zip_longest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.envs.atari import AtariEnv
from src.common.utils import namedarraytuple


OfflineSamples = namedarraytuple("OfflineSamples", ["observation", "action", "reward", "done"])


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
                new_filename = os.path.join(tmp_data_path, Path(os.path.basename(filename)[:-3]+".npy"))
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
                games: List[str],
                checkpoints: List[int],
                frames: int,
                k_step: int,
                max_size: int,
                full_action_set: bool,
                dataset_on_gpu: bool,
                dataset_on_disk: bool) -> None:
        
        self.games = [DQNReplayDataset(data_path,
                        tmp_data_path,
                        game,
                        ckpt,
                        frames,
                        k_step,
                        max_size,
                        full_action_set,
                        dataset_on_gpu,
                        dataset_on_disk) for ckpt in checkpoints for game in games]

        self.num_blocks = len(self.games)
        self.block_len = len(self.games[0])

    def __len__(self) -> int:
        return len(self.games) * len(self.games[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        game_index = index % len(self.games)
        index = index // len(self.games)
        return self.games[game_index][index]


def sanitize_batch(batch: OfflineSamples) -> OfflineSamples:
    has_dones, inds = torch.max(batch.done, 0)
    for i, (has_done, ind) in enumerate(zip(has_dones, inds)):
        if not has_done:
            continue
        batch.observation[ind+1:, i] = batch.observation[ind, i]
        batch.reward[ind+1:, i] = 0
    return batch


def get_offline_dataloaders(
    data_path: Path,
    tmp_data_path: Path,
    games: List[str],
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
    shuffle_checkpoints: bool=False,
    **kwargs) -> DataLoader:
  
    def collate(batch):
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

    dataset = MultiDQNReplayDataset(data_path, 
                                    tmp_data_path, 
                                    games, 
                                    checkpoints, 
                                    frames, 
                                    k_step, 
                                    max_size,
                                    full_action_set, 
                                    dataset_on_gpu, 
                                    dataset_on_disk)

    if shuffle_checkpoints:
        data = get_from_dataloaders(dataset.games)
        shuffled_data = shuffle_batch_dim(*data)
        assign_to_dataloaders(dataset.games, *shuffled_data)

    if group_read_factor != 0:
        sampler = CacheEfficientSampler(dataset.num_blocks, dataset.block_len, group_read_factor)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                collate_fn=collate,
                                drop_last=True,
                                prefetch_factor=prefetch_factor)
    else:
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                collate_fn=collate,
                                drop_last=True,
                                prefetch_factor=prefetch_factor)

    return dataloader


class CacheEfficientSampler(torch.utils.data.Sampler):
    def __init__(self, num_blocks, block_len, num_repeats=20, generator=None):
        self.num_blocks = num_blocks
        self.block_len = block_len  # For now, assume all have same length
        self.num_repeats = num_repeats
        self.generator = generator
        if self.num_repeats == "all":
            self.num_repeats = block_len

    def num_samples(self) -> int:
        # dataset size might change at runtime
        return self.block_len * self.num_blocks

    def __iter__(self):
        n = self.num_samples()
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        self.block_ids = [np.arange(self.num_blocks)] * (self.block_len // self.num_repeats)
        blocks = torch.randperm(n // self.num_repeats, generator=generator) % self.num_blocks
        intra_orders = [torch.randperm(self.block_len, generator=generator) + self.block_len * i for i in
                        range(self.num_blocks)]
        intra_orders = [i.tolist() for i in intra_orders]

        indices = []
        block_counts = [0] * self.num_blocks

        for block in blocks:
            indices += intra_orders[block][
                       (block_counts[block] * self.num_repeats):(block_counts[block] + 1) * self.num_repeats]
            block_counts[block] += 1

        return iter(indices)

    def __len__(self):
        return self.num_samples()


def shuffle_by_trajectory():
    raise NotImplementedError


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def shuffle_batch_dim(observations,
                      rewards,
                      actions,
                      dones,
                      obs_on_disk=True,
                      chunk_num=1):
    """
    :param observations: (T, B, *) obs tensor, optionally mmap
    :param rewards: (T, B) rewards tensor
    :param actions: (T, B, *) actions tensor
    :param dones: (T, B) termination tensor
    :param obs_on_disk: Store observations on disk.  Generally true if using
    more than ~3M transitions
    :return:
    """
    batch_dim = observations[0].shape[1]
    num_sources = len(observations)
    batch_allocations = [np.sort((np.arange(batch_dim) + i) % num_sources) for i in range(num_sources)]

    shuffled_observations, shuffled_rewards, shuffled_actions, shuffled_dones = [], [], [], []

    checkpoints = list(range(num_sources))
    for sources, shuffled, filetype in zip([observations, rewards, actions, dones],
                                           [shuffled_observations, shuffled_rewards, shuffled_actions, shuffled_dones],
                                           ["observations", "rewards", "actions", "dones"]):
        ind_counters = [0]*num_sources

        for start in checkpoints[::chunk_num]:
            chunk = checkpoints[start:start+chunk_num]
            chunk_arrays = []
            for i in chunk:
                if isinstance(sources[0], torch.Tensor):
                    new_array = torch.zeros_like(sources[0])
                else:
                    new_array = np.zeros(sources[0].shape, dtype=sources[0].dtype)
                chunk_arrays.append(new_array)
            for source, allocation in zip(sources, batch_allocations):
                print(chunk, ind_counters)
                for i, new_array in zip(chunk, chunk_arrays):
                    mapped_to_us = [b for b, dest in enumerate(allocation) if dest == i]
                    new_array[:, ind_counters[i]:ind_counters[i]+len(mapped_to_us)] = source[:, mapped_to_us[0]:mapped_to_us[-1]+1]
                    ind_counters[i] += len(mapped_to_us)

            for i, new_array in zip(chunk, chunk_arrays):
                if filetype == "observations" and obs_on_disk:
                    filename = observations[i].filename.replace(".npy", "_shuffled.npy")
                    print("Stored shuffled obs on disk at {}".format(filename))
                    np.save(filename, new_array)
                    del new_array
                    new_array = np.load(filename, mmap_mode="r+")
                shuffled.append(new_array)

    return shuffled_observations, shuffled_rewards, shuffled_actions, shuffled_dones


def get_from_dataloaders(dataloaders):
    observations = [dataloader.observations for dataloader in dataloaders]
    rewards = [dataloader.rewards for dataloader in dataloaders]
    actions = [dataloader.actions for dataloader in dataloaders]
    dones = [dataloader.terminal for dataloader in dataloaders]

    return observations, rewards, actions, dones


def assign_to_dataloaders(dataloaders, observations, rewards, actions, dones):
    for dl, obs, rew, act, done in zip(dataloaders, observations, rewards, actions, dones):
        dl.observations = obs
        dl.rewards = rew
        dl.actions = act
        dl.terminal = done
