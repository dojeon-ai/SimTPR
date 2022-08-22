cd ..
cd ..
python run_atari_pretrain.py \
    --group_name vit_baseline \
    --exp_name act_vit_s_t8 \
    --mode test \
    --config_name mixed_action_vit_s \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 2 \
    --overrides dataloader='mixed_bn256_t8'
