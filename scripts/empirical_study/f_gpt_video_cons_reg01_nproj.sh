cd ..
cd ..
python run_atari_finetune.py \
    --group_name emp_study \
    --exp_name gpt_video_cons_reg01_nproj \
    --mode full \
    --config_name gpt_impala \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides pretrain.artifact_name='atari_pretrain/gpt_video_cons_reg01_nproj:latest'