cd ..
cd ..
python run_atari_pretrain.py \
    --group_name rssm_t_step \
    --exp_name ar_gpt_bs64_t8 \
    --config_name mixed_rssm_impala \
    --mode test \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides dataloader.train.t_step=8 \
    --overrides dataloader.train.batch_size=64 \
    --overrides model.head.dec_strategy='ar'
    