cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline_10 \
    --exp_name trajformer_o \
    --config_name mixed_trajformer_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.dataset_type='trajectory' \
    --overrides trainer.obs_lmbda=1.0 \
    --overrides trainer.act_lmbda=0.0 \
    --overrides trainer.rew_lmbda=0.0 