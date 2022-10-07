cd ..
cd ..
python run_atari_pretrain.py \
    --group_name clt_baseline \
    --exp_name trajformer_oa \
    --config_name mixed_trajformer_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.dataset_type='demonstration'