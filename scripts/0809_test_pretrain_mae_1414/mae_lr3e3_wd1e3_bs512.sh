cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='test_mae_14*14' \
    --overrides exp_name='mae_lr3e3_wd1e3_bn512' \
    --config_name mixed_mae_vit-s \
    --overrides trainer.optimizer.lr=0.006 \
    --overrides trainer.optimizer.weight_decay=0.001 \
    --overrides trainer.scheduler.max_lr=0.006 \
    --overrides trainer.scheduler.min_lr=0.0006 \
    --overrides trainer.update_freq=2
