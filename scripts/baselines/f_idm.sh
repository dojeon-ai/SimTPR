cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name idm_impala \
    --mode full \
    --config_name idm_impala \
    --num_seeds 10 \
    --num_devices 4 \
    --num_exp_per_device 3
