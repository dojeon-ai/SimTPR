cd ..
python run_atari_pretrain.py \
    --group_name test \
    --exp_name gpt_video_cont \
    --config_name mixed_gpt_impala \
    --mode full \
    --debug True \
    --num_seeds 1 \
    --num_devices 2 \
    --num_exp_per_device 1 \
    --overrides trainer.dataset_type='video' \
    --overrides trainer.reg_lmbda=0.0 \
    --overrides trainer.loss_type='cont' \
