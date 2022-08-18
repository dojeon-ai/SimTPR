cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name test_mae_action \
    --mode test \
    --config_name mixed_mae-action_vit-s \
    --num_seeds 1 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/test_mae:v0' \
    --model_path '0/100/model.pth' \
    --overrides trainer.pretrain_type='naive'
