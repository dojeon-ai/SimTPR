cd ..
cd ..
python run_atari_pretrain.py \
    --group_name rssm_t_step \
    --exp_name ar_gpt_bs32_t16 \
    --config_name mixed_rssm_impala \
    --mode test \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides dataloader.train.t_step=16 \
    --overrides dataloader.train.batch_size=32 \
    --overrides model.head.dec_strategy='ar'
    