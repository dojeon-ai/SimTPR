cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name hicat_demon \
    --config_name mixed_hicat_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 8 \
    --num_exp_per_device 1 \
    --overrides trainer.state_lmbda=0.0 \
    --overrides trainer.demon_lmbda=1.0 \
    --overrides trainer.traj_lmbda=0.0 \
