from src.datasets.atari_dataset import get_offline_dataloaders

if __name__ == '__main__':
    dataloader = get_offline_dataloaders(
        data_path='/home/nas3_userK/hojoonlee/projects/video_rl/data/atari',
        tmp_data_path='/home/nas3_userK/hojoonlee/projects/video_rl/data/atari',
        games=['Pong'],
        checkpoints=[1,25],
        frames=4,
        k_step=10, # length of the future trajectory to predict
        max_size=1000000,
        dataset_on_gpu=False,
        dataset_on_disk=True,
        batch_size=2,
        full_action_set=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=8,
        group_read_factor=0,
        shuffle_checkpoints=False)

    for batch in dataloader:
        import pdb
        pdb.set_trace()