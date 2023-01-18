cd ..
cd ..
python run_atari_pretrain.py \
    --group_name baseline \
    --exp_name gpt_cons_demon_nproj_reg0 \
    --config_name mixed_gpt_impala \
    --mode full \
    --debug False \
    --num_seeds 1 \
    --num_devices 4 \
    --num_exp_per_device 1 \
    --overrides trainer.dataset_type='demonstration' \
    --overrides trainer.reg_lmbda=0.0 \
    --overrides model.head.proj_bn='False' \
    --overrides trainer.act_lmbda=1.0 \