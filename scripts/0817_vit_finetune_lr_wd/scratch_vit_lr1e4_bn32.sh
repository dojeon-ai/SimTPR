
cd ..
cd ..
python run_atari_finetune.py \
    --group_name test_vit_finetune \
    --exp_name vit_lr1e4_bn32 \
    --mode test \
    --config_name drq_vit \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3
