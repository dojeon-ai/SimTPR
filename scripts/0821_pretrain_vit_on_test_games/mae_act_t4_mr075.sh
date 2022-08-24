cd ..
cd ..
python run_atari_pretrain.py \
    --group_name vit_baseline \
    --exp_name mae_act_vit_s_t4_mr075 \
    --mode test \
    --config_name mixed_action_vit_s \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 2 \
    --overrides dataloader='mixed_bn256_t4' \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/mae_vit_s_t4_mr075:v0' \
    --model_path '0/100/model.pth' \
    --overrides trainer.pretrain_type='naive'
