cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='mae-action_lr_wd' \
    --overrides exp_name='mae-action_lr1e4_wd1e4' \
    --config_name mixed_mae-action_vit-s \
    --overrides trainer.optimizer.lr=0.0001 \
    --overrides trainer.optimizer.weight_decay=0.0001 \
    --overrides trainer.scheduler.max_lr=0.0001 \
    --overrides trainer.scheduler.min_lr=0.00001 
