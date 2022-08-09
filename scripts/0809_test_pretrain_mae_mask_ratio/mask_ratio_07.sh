cd ..
cd ..
python run_atari_pretrain.py \
    --group_name vit_mask_ratio \
    --exp_name mask_ratio_07 \
    --mode test \
    --config_name mixed_mae_vit-s \
    --num_seeds 1 \
    --num_devices 2 \
    --num_exp_per_device 2 \
    --overrides trainer.patch_mask_ratio=0.7
    