
cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name mae_idm_vit_d192_l4 \
    --mode test \
    --config_name drq_vit \
    --num_seeds 10 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --overrides agent.optimizer.lr=0.0003 \
    --overrides agent.finetune_type='naive' \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/mae_idm_d192_l4:v0' \
    --model_path '0/100/model.pth'