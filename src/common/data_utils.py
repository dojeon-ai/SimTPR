import torch
import numpy as np
from src.common.class_utils import namedarraytuple

OfflineSamples = namedarraytuple("OfflineSamples", ["observation", "action", "reward", "done", "rtg"])

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
                if filetype in ["observations"] and obs_on_disk:
                    filename = observations[i].filename.replace(".npy", "_shuffled.npy")
                    print("Stored shuffled obs on disk at {}".format(filename))
                    np.save(filename, new_array)
                    del new_array
                    new_array = np.load(filename, mmap_mode="r+")
                shuffled.append(new_array)

    return shuffled_observations, shuffled_rewards, shuffled_actions, shuffled_dones


def sanitize_batch(batch: OfflineSamples) -> OfflineSamples:
    done_idx = torch.nonzero(batch.done==1)

    # done is tracked in reverse-order to keep consistency with evaluation stage.
    for idx in done_idx:
        row = idx[0]
        col = idx[1]
        batch.observation[row, :col] = 0
        batch.reward[row, :col] = 0
        batch.rtg[row, :col] = 0
        
    return batch

