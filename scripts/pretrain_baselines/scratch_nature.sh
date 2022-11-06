cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name nature \
    --config_name mixed_scratch_nature \
    --mode full \
    --debug True \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 
