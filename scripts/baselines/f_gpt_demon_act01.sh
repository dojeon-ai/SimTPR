cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name gpt_demon_act01 \
    --mode full \
    --config_name gpt_impala \
    --num_seeds 10 \
    --num_devices 8 \
    --num_exp_per_device 3 \
    --overrides pretrain.artifact_name='atari_pretrain/gpt_demon_act01:latest'