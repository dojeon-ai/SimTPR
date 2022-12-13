cd ..
cd ..
python run_pretrain.py --config_name mixed_bert_impala \
                       --overrides trainer.mask_ratio=0.3 \

python run_pretrain.py --config_name mixed_bert_impala \
                       --overrides trainer.mask_ratio=0.5 \

python run_pretrain.py --config_name mixed_gpt_impala \
                       --overrides trainer.aug_types=[] \
                       
python run_pretrain.py --config_name mixed_barlow_impala \
                       
                       

                        