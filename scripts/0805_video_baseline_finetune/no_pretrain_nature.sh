cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name no_pretrain_nature \
    --mode full \
    --config_name drq_nature \
    --num_seeds 10 \
    --num_devices 4 \
    --num_exp_per_device 3 \
