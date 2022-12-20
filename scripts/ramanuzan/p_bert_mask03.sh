cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name bert_mask03_nproj \
    --config_name mixed_bert_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.mask_ratio=0.3 \
    --overrides model.head.proj_bn='False'
