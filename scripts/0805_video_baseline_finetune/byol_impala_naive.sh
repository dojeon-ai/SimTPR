cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name byol_impala \
    --mode full \
    --config_name drq_impala \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/byol_impala:v0' \
    --model_path '0/10/model.pth'
