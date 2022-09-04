cd ..
cd ..
python run_atari_pretrain.py \
    --group_name test \
    --exp_name store_flow \
    --mode 3 \
    --config_name mixed_mae_vit \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
