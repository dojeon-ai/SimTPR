cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name trajformer_demon_cont_lmbda08 \
    --config_name mixed_trajformer_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.dataset_type='demonstration' \
    --overrides trainer.obs_loss_type='contrastive' \
    --overrides trainer.temperature=0.1 \
    --overrides trainer.obs_lmbda=0.8 \
    --overrides trainer.act_lmbda=0.2