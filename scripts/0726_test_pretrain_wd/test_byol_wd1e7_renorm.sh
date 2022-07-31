cd ..
cd ..
python run_atari_pretrain.py --exp_name byol_wd1e7_renorm --mode test --config_name mixed_byol_impala --num_seeds 1 --num_devices 4 --num_exp_per_device 1 --overrides trainer.optimizer.weight_decay=0.0000001 --overrides trainer.num_epochs=16 --overrides model.backbone.renormalize=True

