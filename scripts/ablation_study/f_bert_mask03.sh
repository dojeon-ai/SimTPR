cd ..
cd ..
python run_atari_finetune.py \
    --group_name abl_study \
    --exp_name bert_mask03_nproj_impala \
    --mode full \
    --config_name bert_impala \
    --num_seeds 3 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides pretrain.artifact_name='atari_pretrain/bert_mask03_nproj:latest'