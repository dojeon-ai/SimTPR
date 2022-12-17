cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name gpt_video_cons_reg0001_impala \
    --mode full \
    --config_name gpt_impala \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --artifact_name='atari_pretrain/gpt_video_cons_reg0001:latest'