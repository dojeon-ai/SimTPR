cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name bcq_impala \
    --mode full \
    --config_name bcq_impala \
    --num_seeds 10 \
    --num_devices 4 \
    --num_exp_per_device 3
