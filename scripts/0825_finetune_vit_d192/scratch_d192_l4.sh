
cd ..
cd ..
python run_atari_finetune.py \
    --group_name vit_finetune \
    --exp_name scratch_vit_lr3e4_d192_l4 \
    --mode test \
    --config_name drq_vit \
    --num_seeds 5 \
    --num_devices 4 \
    --num_exp_per_device 2 \
    --overrides agent.optimizer.lr=0.0003 \
    --overrides model='vit_d192'