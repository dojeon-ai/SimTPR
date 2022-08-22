
cd ..
cd ..
python run_atari_finetune.py \
    --group_name vit_finetune_d256 \
    --exp_name act_vit_t4_lr3e4 \
    --mode test \
    --config_name drq_vit \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides agent.env='atari_t4' \
    --overrides agent.finetune_type='naive' \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/act_vit_s_t4:v0' \
    --model_path '0/50/model.pth'
