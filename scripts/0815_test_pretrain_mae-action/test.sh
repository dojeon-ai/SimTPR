cd ..
cd ..
python run_pretrain.py \
    --config_name mixed_mae-action_vit-s \
    --overrides group_name='test' \
    --overrides exp_name='test' \
    --overrides use_artifact=True \
    --overrides artifact_name='mask_ratio_05:v1' \
    --overrides model_path='Alien/0/100/model.pth' \
    --overrides dataloader.game='Alien' \
    --overrides env.game='alien' \
    --overrides trainer.pretrain_type='freeze'    