cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name bc \
    --config_name mixed_bc_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1