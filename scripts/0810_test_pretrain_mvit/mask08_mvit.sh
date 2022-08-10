cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_motion_mvit-s \
    --overrides group_name='test' \
    --overrides exp_name='test' \
    --overrides use_artifact=True \
    --overrides artifact_name= \
    --overrides model_path= \