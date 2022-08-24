
cd ..
cd ..
python run_atari_finetune.py \
    --group_name vit_finetune_d256 \
    --exp_name scratch_vit_t4_lr3e4 \
    --mode test \
    --config_name drq_vit \
    --num_seeds 4 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides env='atari_t4' \
    --overrides model='vit_t4'
