cd ..
cd ..
python run_atari_finetune.py \
    --group_name test_finetune \
    --exp_name byol_reduced_01 \
    --mode test \
    --config_name drq_impala \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/byol_wd1e7_renorm:v0' \
    --model_path '0/4/model.pth' \
    --overrides 'agent.finetune_type=reduced' \
    --overrides 'agent.backbone_lr_scale=0.1'