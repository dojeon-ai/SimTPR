cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name scratch_action_vit_s \
    --mode test \
    --config_name mixed_action_vit_s \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
