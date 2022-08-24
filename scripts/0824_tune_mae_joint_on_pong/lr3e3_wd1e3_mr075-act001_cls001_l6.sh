cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae_vit-s \
    --overrides group_name='mae_joint_lr_wd' \
    --overrides exp_name='mae_lr3e3_wd1e3_mr75_act001_cls001_l6' \
    --config_name mixed_mae_joint_vit \
    --overrides trainer.patch_mask_ratio=0.75 \
    --overrides trainer.act_lmbda=0.03 \    
    --overrides trainer.cls_lmbda=0.0  \
    --overrides model.backbone.enc_depth=6 \    