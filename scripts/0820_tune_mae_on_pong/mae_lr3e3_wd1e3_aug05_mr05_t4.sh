cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='mae_lr_wd' \
    --overrides exp_name='mae_lr3e3_wd1e3_t4_aug05_mr05' \
    --config_name mixed_mae_vit_s \
    --overrides dataloader='mixed_bn256_t4' \
    --overrides trainer.patch_mask_ratio=0.5