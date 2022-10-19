cd ..
cd ..
python run_atari_pretrain.py \
    --group_name tune_rssm \
    --exp_name model_fx_rssm_lr1e3_idm0 \
    --config_name mixed_rssm_impala \
    --mode test \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.optimizer.lr=0.001 \
    --overrides trainer.scheduler.max_lr=0.001 \
    --overrides trainer.scheduler.min_lr=0.0001 \
    --overrides trainer.idm_lmbda=0 \