cd ..
cd ..
python run_atari_pretrain.py \
    --group_name simclr_tstep \
    --exp_name simclr_tstep16 \
    --mode test \
    --config_name mixed_simclr_impala \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.num_epochs=32 \
    --overrides dataloader.t_step=16

