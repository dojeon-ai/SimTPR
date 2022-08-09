cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='test_mae_14*14_lmbda' \
    --overrides exp_name='mae_lmbda0.3' \
    --config_name mixed_mae_vit-s \
    --overrides trainer.act_mask_ratio=0.2 \
    --overrides trainer.lmbda=0.3 