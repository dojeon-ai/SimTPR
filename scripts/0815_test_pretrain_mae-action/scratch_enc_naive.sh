cd ..
cd ..
python run_atari_pretrain.py \
    --group_name test_mae-action \
    --exp_name scratch_enc_naive \
    --mode test \
    --config_name mixed_mae-action_vit-s \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
