cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline_10 \
    --exp_name curl \
    --config_name mixed_atc_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 