cd ..
cd ..
python run_atari_finetune.py --exp_name drq_impala_renorm --mode test --config_name drq_impala --num_seeds 3 --num_devices 4 --num_exp_per_device 3 --overrides model.backbone.renormalize=True
