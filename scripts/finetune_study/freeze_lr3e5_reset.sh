cd ..
cd ..
python run_atari_finetune.py \
    --group_name ft_study \
    --exp_name gpt_demon_freeze_lr3e5_reset25000 \
    --mode full \
    --config_name gpt_impala \
    --num_seeds 3 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --overrides pretrain.artifact_name='atari_pretrain/gpt_demon_act1:latest' \
    --overrides agent.reset_freq=25000