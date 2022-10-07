cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name trajformer_demon_lr1e4 \
    --config_name mixed_trajformer_impala \
    --mode test \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.dataset_type='demonstration' \
    --overrides trainer.optimizer.lr=0.0001 \
    --overrides trainer.scheduler.max_lr=0.0001 \
    --overrides trainer.scheduler.min_lr=0.0001