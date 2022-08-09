cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='test_mae_12*12' \
    --overrides exp_name='mae_lr3e4_wd1e4' \
    --config_name mixed_mae_vit-s \
    --overrides trainer.optimizer.lr=0.0003 \
    --overrides trainer.optimizer.weight_decay=0.0001 \
    --overrides trainer.scheduler.max_lr=0.0003 \
    --overrides trainer.scheduler.min_lr=0.00003 
