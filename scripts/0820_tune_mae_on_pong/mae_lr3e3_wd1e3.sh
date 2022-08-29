cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='mae_lr_wd' \
    --overrides exp_name='mae_lr3e3_wd1e3_t8_aug05' \
    --config_name mixed_mae_vit_s \
    --overrides dataloader='mixed_bn256_t8'
