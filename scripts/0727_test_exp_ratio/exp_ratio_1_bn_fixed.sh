cd ..
cd ..
python run_atari_finetune.py \
    --group_name test_exp_ratio \
    --exp_name exp_ratio_1_bn_fixed \
    --mode test \
    --config_name drq_impala \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides model.backbone.expansion_ratio=1
