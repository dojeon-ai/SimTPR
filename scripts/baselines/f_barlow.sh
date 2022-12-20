cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name barlow_impala \
    --mode full \
    --config_name barlow_impala \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides pretrain.artifact_name='atari_pretrain/barlow:latest'