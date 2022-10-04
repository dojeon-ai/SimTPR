cd ..
cd ..
python run_atari_pretrain.py \
    --group_name rssm_decoder \
    --exp_name dec_64*64_conv \
    --config_name mixed_rssm_impala \
    --mode test \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides model.head.dec_type='conv_det'