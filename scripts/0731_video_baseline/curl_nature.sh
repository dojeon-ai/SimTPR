cd ..
cd ..
python run_atari_pretrain.py \
    --group_name video_baseline \
    --exp_name curl_nature \
    --mode full \
    --config_name mixed_curl_nature \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.num_epochs=10 \
    --overrides dataloader.t_step=1

