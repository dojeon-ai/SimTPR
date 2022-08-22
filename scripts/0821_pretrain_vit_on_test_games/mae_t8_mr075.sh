cd ..
cd ..
python run_atari_pretrain.py \
    --group_name vit_baseline \
    --exp_name mae_vit_s_t8_mr075 \
    --mode test \
    --config_name mixed_mae_vit_s \
    --num_seeds 1 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --overrides dataloader='mixed_bn256_t8' \
    --overrides trainer.patch_mask_ratio=0.75 