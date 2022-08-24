
cd ..
cd ..
python run_atari_finetune.py \
    --group_name vit_finetune_d256 \
    --exp_name mae_action_vit_lr3e4_cls_last \
    --mode test \
    --config_name drq_vit \
    --num_seeds 6 \
    --num_devices 8 \
    --num_exp_per_device 2 \
    --overrides agent.optimizer.lr=0.0003 \
    --overrides agent.finetune_type='naive' \
    --overrides model.backbone.pool='cls_last' \
    --overrides model.policy.in_features=256 \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/mae_act_vit_s_t8_mr075:v0' \
    --model_path '0/50/model.pth' 
