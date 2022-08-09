cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='test_mae' \
    --overrides exp_name='mae_lr1e3_wd1e3_drop0' \
    --config_name mixed_mae_vit-s \
    --overrides trainer.optimizer.lr=0.001 \
    --overrides trainer.optimizer.weight_decay=0.001 \
    --overrides trainer.scheduler.max_lr=0.001 \
    --overrides trainer.scheduler.min_lr=0.0001 
