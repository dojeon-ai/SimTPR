cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name no_pretrain_impala_orthogonal \
    --mode full \
    --config_name drq_impala \
    --num_seeds 10 \
    --num_devices 8 \
    --num_exp_per_device 2 \
    --overrides model.backbone.init_type='orthogonal'
