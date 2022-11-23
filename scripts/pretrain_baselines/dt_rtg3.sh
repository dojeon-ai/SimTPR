cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name dt_rtg3 \
    --config_name mixed_dt_impala \
    --mode full \
    --debug True \
    --num_seeds 1 \
    --num_devices 2 \
    --num_exp_per_device 1 \
    --overrides trainer.max_rtg_ratio=3.0 \
    --overrides pretrain=dt 