cd ..
cd ..
python run_atari_pretrain.py \
    --group_name test_mae-action \
    --exp_name mae_enc_freeze \
    --mode test \
    --config_name mixed_mae-action_vit-s \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/mask_ratio_05:v1' \
    --model_path '0/100/model.pth' \
    --overrides trainer.pretrain_type='freeze'
