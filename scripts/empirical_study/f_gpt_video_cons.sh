cd ..
cd ..
python run_atari_finetune.py \
    --group_name emp_study \
    --exp_name gpt_video_cons_impala \
    --mode full \
    --config_name gpt_impala \
    --num_seeds 10 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides pretrain.artifact_name='atari_pretrain/gpt_video_cons:latest'