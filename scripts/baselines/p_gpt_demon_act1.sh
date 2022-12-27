cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name gpt_demon_act1 \
    --config_name mixed_gpt_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --overrides trainer.dataset_type='demonstration' \
    --overrides trainer.act_lmbda=1.0 \
