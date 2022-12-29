cd ..
cd ..
python run_atari_finetune.py \
    --group_name ft_study \
    --exp_name gpt_video_reset_tbb-train_lr3e5_reset20000 \
    --mode full \
    --config_name gpt_impala \
    --num_seeds 3 \
    --num_devices 8 \
    --num_exp_per_device 2 \
    --overrides pretrain.artifact_name='atari_pretrain/gpt_video_cons_reg001_nproj:latest' \
    --overrides agent.finetune_type='naive' \
    --overrides agent.eval_backbone_mode='eval' \
    --overrides agent.train_backbone_mode='train' \
    --overrides agent.train_target_backbone_mode='eval' \
    --overrides agent.optimizer.lr=0.00003 \
    --overrides agent.reset_freq=20000