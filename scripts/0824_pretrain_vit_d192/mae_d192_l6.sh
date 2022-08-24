cd ..
cd ..
python run_atari_pretrain.py \
    --group_name vit_baseline \
    --exp_name mae_d192_l6 \
    --mode test \
    --config_name mixed_mae_vit \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 2 \
    --overrides trainer.patch_mask_ratio=0.75 \
    --overrides model.backbone.enc_depth=6