cd ..
cd ..
python run_atari_finetune.py \
    --group_name test_finetune \
    --exp_name no_pretrain \
    --mode test \
    --config_name drq_impala \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 
