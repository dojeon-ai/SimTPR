
cd ..
cd ..
python run_atari_finetune.py \
    --group_name vit_finetune_d256 \
    --exp_name mae_action_vit_lr3e4 \
    --mode test \
    --config_name drq_vit \
    --num_seeds 3 \
    --num_devices 8 \
    --num_exp_per_device 3 \
    --overrides agent.optimizer.lr=0.0003 \
    --overrides agent.finetune_type='naive' \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/mae_act_vit_s_t8_mr075:v0' \
    --model_path '0/50/model.pth'
