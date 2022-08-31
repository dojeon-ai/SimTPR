
cd ..
cd ..
python run_atari_finetune.py \
    --group_name vit_finetune \
    --exp_name mae_idm_vit_lr3e4_d192_l4_gc025 \
    --mode test \
    --config_name drq_vit \
    --num_seeds 5 \
    --num_devices 4 \
    --num_exp_per_device 2 \
    --overrides agent.optimizer.lr=0.0003 \
    --overrides model='vit_d192' \
    --overrides agent.finetune_type='naive' \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/mae_idm_d192_l4:v0' \
    --model_path '0/100/model.pth' \
    --overrides agent.clip_grad_norm=0.25