cd ..
cd ..
python run_atari_pretrain.py \
    --group_name vit_baseline \
    --exp_name idm_d192_l4 \
    --mode test \
    --config_name mixed_idm_vit \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 2