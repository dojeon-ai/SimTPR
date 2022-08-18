cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name test_mae \
    --mode test \
    --config_name mixed_mae-patch_vit-s \
    --num_seeds 1 \
    --num_devices 8 \
    --num_exp_per_device 1    