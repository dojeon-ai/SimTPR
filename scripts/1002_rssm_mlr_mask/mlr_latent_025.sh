cd ..
cd ..
python run_atari_pretrain.py \
    --group_name rssm_mlr_mask \
    --exp_name mlr_latent_025 \
    --config_name mixed_mlr_impala \
    --mode test \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.mask_type='latent' \
    --overrides trainer.mask_ratio=0.25