cd ..
cd ..
python run_atari_pretrain.py \
    --group_name vit_baseline \
    --exp_name mae_d128_l8 \
    --mode test \
    --config_name mixed_mae_vit \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 2 \
    --overrides trainer.patch_mask_ratio=0.75 \
    --overrides model='vit-tiny_mae'