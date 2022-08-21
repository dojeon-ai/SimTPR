
cd ..
cd ..
python run_atari_finetune.py \
    --group_name test_vit_finetune \
    --exp_name scratch_action_naive_lr3e4 \
    --mode test \
    --config_name drq_vit \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides agent.optimizer.lr=0.0003 \
    --overrides agent.finetune_type='naive' \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/scratch_action_vit_s:v0' \
    --model_path '0/50/model.pth'
