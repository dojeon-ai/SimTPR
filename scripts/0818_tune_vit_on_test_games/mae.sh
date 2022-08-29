cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name mae_vit_s \
    --mode test \
    --config_name mixed_mae_vit_s \
    --num_seeds 1 \
    --num_devices 8 \
    --num_exp_per_device 1    