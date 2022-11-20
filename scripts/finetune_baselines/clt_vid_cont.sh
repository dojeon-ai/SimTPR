cd ..
cd ..
python run_atari_finetune.py \
    --group_name baseline \
    --exp_name trajformer_vid_cont_impala \
    --mode full \
    --config_name trajformer_impala \
    --num_seeds 10 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --overrides pretrain.artifact_name='atari_pretrain/trajformer_vid_cont:latest' \
