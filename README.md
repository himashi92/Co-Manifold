# Co-Manifold
This repo contains the supported pytorch code and configuration files to reproduce the results of the Co-Manifold Learning for Semi-supervised Medical Image Segmentation Article.

## Abstract

We investigate jointly learning hyperbolic and Euclidean space representations and match the consistency for semi-supervised medical image segmentation. We argue that for complex medical volumetric data, hyperbolic spaces are better suited to model data inductive biases. We propose an approach incorporating the two geometries to co-train an encoder-decoder model with a Hyperbolic probabilistic latent space and a separate encoder-decoder model with a Euclidean probabilistic latent space with complementary representations, thereby bridging the gap of co-training across manifolds (Co-Manifold learning) in a principled manner. To capture complementary information and hierarchical relationships, we propose a latent space embedding loss based on cosine similarity, which acts as a disagreement loss across manifolds. Further, we incorporate adversarial learning to enhance segmentation performance, and by doing so, we propose dual uncertainty-aware loss to enhance the learning of confident regions from one another. Our proposed method achieves competitive results on multiple benchmarks for semi-supervised medical image segmentation on medical scans.

## Link to full paper:
To be Added

## Proposed Architecture
![Proposed Architecture](img/Co-Manifold.png?raw=true)

## System requirements
Under this section, we provide details on environmental setup and dependencies required to train/test the Co-BioNet model.
This software was originally designed and run on a system running Ubuntu (Compatible with Windows 11 as well).
<br>
All the experiments are conducted on Ubuntu 20.04 Focal version with Python 3.8.
<br>
To train Co-Manifold with the given settings, the system requires a GPU with at least 40GB. All the experiments are conducted on Nvidia A40 single GPU.
(Not required any non-standard hardware)
<br>

### Create a virtual environment

```bash 
pip install virtualenv
virtualenv -p /usr/bin/python3.8 venv
source venv/bin/activate
```

### Installation guide 

- Install torch : 
```bash
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
- Install other dependencies :
```bash 
pip install -r requirements.txt
```

## Dataset Preparation
The experiments are conducted on two publicly available datasets,
- 2018 Left Atrial Segmentation Challenge Dataset : http://atriaseg2018.cardiacatlas.org
- MSD BraTS Dataset : http://medicaldecathlon.com/

Pre-processed data can be found in folder data.

## Trained Model Weights
Download trained model weights from this shared drive [link](https://drive.google.com/drive/folders/1n8XREomBOg9hiU59Bn5BW54sojNxApcw?usp=share_link), and put it under folder **code_la/model_weights** or **code_brats/model_weights**

## Train Model
- To train the model for LA MRI dataset on 10% Lableled data
```bash
cd code_la
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name "LA" --labelnum 8 --dl_w 0.8 --ce_w 0.8 --alpha 0.005 --beta 0.02 --t_m 0.2 --hidden-dim 256 --batch_size 6 --labeled_bs 3 &> la_10.out &
```

- To train the model for LA MRI dataset on 20% Lableled data
```bash
cd code_la
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name "LA" --labelnum 16 --dl_w 0.8 --ce_w 0.8 --alpha 0.005 --beta 0.02 --t_m 0.2 --hidden-dim 256 --batch_size 6 --labeled_bs 3 &> la_20.out &
```

- To train the model for MSD BraTS MRI dataset on 10% Lableled data
```bash
cd code_brats
nohup python train.py --dataset_name MSD_BRATS --labelnum 39 --dl_w 1.0 --ce_w 1.0 --alpha 0.005 --beta 0.02 --t_m 0.2 --hidden-dim 256 --batch_size 6 --labeled_bs 3 &> msd_10_perc.out &
```

- To train the model for MSD BraTS MRI dataset on 20% Lableled data
```bash
cd code_brats
nohup python train.py --dataset_name MSD_BRATS --labelnum 77 --dl_w 1.0 --ce_w 1.0 --alpha 0.005 --beta 0.02 --t_m 0.2 --hidden-dim 256 --batch_size 6 --labeled_bs 3 &> msd_20_perc.out &
```
## Test Model

- To test the Co-Manifold ensemble model for LA MRI dataset on 10% Lableled data
```bash
cd code
CUDA_VISIBLE_DEVICES=0 nohup python inference.py --dataset_name "LA" --labelnum 8 --dl_w 0.8 --ce_w 0.8 --alpha 0.005 --beta 0.02 --t_m 0.2 --hidden-dim 256 --exp Co_Manifold --batch_size 6 --labeled_bs 3 &> la_10_eval.out &
```

## Acknowledgements

This repository makes liberal use of code from [capturing-implicit-hierarchical-structure](https://github.com/its-gucci/capturing-implicit-hierarchical-structure) and [MC-Net](https://github.com/ycwu1997/MC-Net/)

## Citing Co-Manifolds

If you find this repository useful, please consider giving us a star ⭐ 

