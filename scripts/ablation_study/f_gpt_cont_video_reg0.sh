cd ..
cd ..
python run_atari_finetune.py \
    --group_name abl_study \
    --exp_name gpt_cont_video_reg0 \
    --mode full \
    --config_name gpt_impala \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides pretrain.artifact_name='atari_pretrain/gpt_video_cont_reg:latest'