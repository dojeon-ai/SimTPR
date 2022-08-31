cd ..
cd ..
python run_atari_pretrain.py \
    --group_name vit_baseline \
    --exp_name mae_idm_vit_d192_l4 \
    --mode test \
    --config_name mixed_idm_vit \
    --num_seeds 1 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/mae_d192_l4:v0' \
    --model_path '0/200/model.pth' \
    --overrides trainer.pretrain_type='naive'
