# On the Importance of Feature Decorrelation for <br> Unsupervised Representation Learning for RL

This repository contains a PyTorch implementation of the paper 

On the Importance of Feature Decorrelation for URL for RL., Lee et al., ICML 2023. 

[paper](https://arxiv.org/abs/2306.05637)
[slide](https://drive.google.com/file/d/1EQunG-JCFufFDSEJv6KrOfOMr85CYT1c/view?usp=sharing)


## Requirements
We assume you have access to a GPU that can run CUDA 11.1 and CUDNN 8. 
Then, the simplest way to install all required dependencies is to create an anaconda environment by running

```
conda env create -f requirements.yml
pip install hydra-core --upgrade
pip install opencv-python
```

After the instalation ends you can activate your environment with
```
conda activate simtpr
```

## Installing Atari environment

### Download Rom dataset
```
python
import urllib.request
urllib.request.urlretrieve('http://www.atarimania.com/roms/Roms.rar','Roms.rar')
```

### Connect Rom dataset to atari_py library
```
apt-get install unrar
unrar x Roms.rar
mkdir rars
mv HC\ ROMS rars
mv ROMS rars
python -m atari_py.import_roms rars
``` 

## Instructions

### Pre-training

If you want to pre-train SimTPR from your own, you first need to download the DQN replay dataset.

```
cd data
bash download_atari_replay_dataset.sh
```

After you download the dataset, you can pretrain the model as

```
python run_pretrain.py --config_name simtpr
```
This will train the SimTPR from the state dataset.

If you would like to train the SimTPR from the demonstration dataset, you can run the code as
```
python run_pretrain.py --config_name simtpr --overrides trainer.dataset_type='demonstration'
```

The pretrained model will be automatically saved in wandb directory, and you can use the pretrained encoder to finetune the policy.


### Fine-tuning
To run fine-tuning from scratch, use the `run_finetune.py` script
```
python run_finetune.py --config_name drq
```

To run fine-tuning from the pre-trained model, you need to perform two steps.

**(1) Download pretrained model**

Download the pretrained models from the below link:

[link](https://drive.google.com/drive/folders/1FlMZkoGiOL-tOfLliHMaVxl6FAUKKOqA?usp=sharing). This will download the prtrained SimTPR on the state dataset (Table 2).

**(2) Finetune from pretrained model**

To finetune the model from the pretrained encoder, simply change the config name.

```
python run_finetune.py --config_name simtpr
```

If you would like to run on different game, select the games such as `['assault', 'breakout', 'pong'. 'qbert']`
```
python run_finetune.py --config_name simtpr --overrides env.game='pong'
```

## Citations

```
@article{lee2023simtpr,
  title={On the Importance of Feature Decorrelation for Unsupervised Representation Learning for Reinforcement Learning},
  author={Hojoon Lee and Koanho Lee and Dongyoon Hwang and Hyunho Lee and Byungkun Lee and Jaegul Choo},
  journal={ICML},
  year={2023}
}
```

## Contact

For personal communication related to SimTPR, please contact Hojoon Lee (`joonleesky@kaist.ac.kr`).
