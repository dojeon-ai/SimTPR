cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name mae_action_vit_s \
    --mode test \
    --config_name mixed_action_vit_s \
    --num_seeds 1 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/test_mae:v0' \
    --model_path '0/100/model.pth' \
    --overrides trainer.pretrain_type='naive'
