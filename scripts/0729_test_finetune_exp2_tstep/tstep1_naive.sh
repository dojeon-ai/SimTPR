cd ..
cd ..
python run_atari_finetune.py \
    --group_name test_finetune_tstep \
    --exp_name tstep1_naive \
    --mode test \
    --config_name drq_impala \
    --num_seeds 6 \
    --num_devices 4 \
    --num_exp_per_device 3 \
    --use_artifact True \
    --artifact_name 'draftrec/atari_pretrain/simclr_tstep1:v0' \
    --model_path '0/32/model.pth'
