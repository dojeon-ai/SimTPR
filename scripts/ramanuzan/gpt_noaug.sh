cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name gpt_no_aug \
    --config_name mixed_gpt_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.aug_types=[] \