cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='mae_lr_wd' \
    --overrides exp_name='mae_lr3e3_wd1e4_t8_aug05' \
    --config_name mixed_mae_vit_s \
    --overrides trainer.optimizer.lr=0.003 \
    --overrides trainer.optimizer.weight_decay=0.0001 \
    --overrides trainer.scheduler.max_lr=0.003 \
    --overrides trainer.scheduler.min_lr=0.0003 \
    --overrides dataloader='mixed_bn256_t8'
