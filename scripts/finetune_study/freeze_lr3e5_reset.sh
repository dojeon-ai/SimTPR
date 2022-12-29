cd ..
cd ..
python run_atari_finetune.py \
    --group_name ft_study \
    --exp_name gpt_video_freeze_lr3e5_reset20000 \
    --mode full \
    --config_name gpt_impala \
    --num_seeds 3 \
    --num_devices 8 \
    --num_exp_per_device 2 \
    --overrides pretrain.artifact_name='atari_pretrain/gpt_video_cons_reg001_nproj:latest' \
    --overrides agent.reset_freq=20000