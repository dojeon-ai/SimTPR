cd ..
cd ..
python run_atari_pretrain.py --exp_name simclr_wd1e6_renorm --mode test --config_name mixed_simclr_impala --num_seeds 1 --num_devices 4 --num_exp_per_device 1 --overrides trainer.optimizer.weight_decay=0.000001 --overrides trainer.num_epochs=16 --overrides model.backbone.renormalize=True

