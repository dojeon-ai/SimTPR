cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name clt_vid_cont \
    --config_name mixed_clt_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.dataset_type='video' \
    --overrides trainer.obs_loss_type='contrastive' \
    --overrides trainer.temperature=0.1 \
    --overrides trainer.projection=True