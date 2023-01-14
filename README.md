# Convolution Sliced Wasserstein
Python3 implementation of the papers [Revisiting Sliced Wasserstein on Images: From Vectorization to Convolution](https://arxiv.org/abs/2204.01188)


Details of the model architecture and experimental results can be found in our papers.

```
@article{nguyen2022revisting,
  title={Revisiting Sliced Wasserstein on Images: From Vectorization to Convolution},
  author={Khai Nguyen and Nhat Ho},
  journal={Advances in Neural Information Processing Systems},
  year={2022},
  pdf={https://arxiv.org/pdf/2204.01188.pdf},
  code={https://github.com/UT-Austin-Data-Science-Group/CSW}
}
```
Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by [Khai Nguyen](https://khainb.github.io). README is on updating process



## Requirement
The code is implemented with Python (3.8.8) and Pytorch (1.9.0).

## What is included?
* (Convolution) Sliced Wasserstein Generator
* Convolution Slicers

## (Convolution) Sliced Wasserstein Generator
### Code organization
* cfg.py : this file contains arguments for training.
* datasets.py : this file implements dataloaders
* functions.py : this file implements training functions
* slicers.py : this file implements slicers for sliced Wasserstein
* trainsw.py : this file is the main file for running.
* models : this folder contains neural networks architecture
* utils : this folder contains implementation of fid score and Inception score
* fid_stat : this folder contains statistic files for fID score.
### Main path arguments
* --slice_type : type of slicers {"sw","gsw","csw","csws","cswd","ncsw","ncsws","ncswd"}
* --dataset : type of dataset {"cifar10","stl10","celeba","celebahq"}
* --bottom_width : "3" for "stl10" and "4" for other datasets.
* --img_size : size of images
* --dis_bs : size of mini-batches
* --model : "sngan_{dataset}"
* --eval_batch_size : batchsize for computing FID
* --L : the number of projections
### Script examples
Train csw (base) on cifar10
```
python trainsw.py \
-gen_bs 128 \
-dis_bs 128 \
--dataset cifar10 \
--img_size 32 \
--max_iter 50000 \
--model sngan_cifar10 \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--exp_name sngan_cifar10 \
--sliced_type csw
```
### Max-CSW, and CPRW

Please use trainmaxsw.py, trainprw.py with similar arguments. The additional arguments include 

* --k : subspace dimension
* --s_lr : learning rate for the max vector (subspace) 
* --s_max_iter : number of updates for the max vector (subspace)

### Pretrained models for SW, CSW-b, CSW-s, CSW-d on CelebA
https://drive.google.com/file/d/1RILLG5ob8LQ4lMODYBAcd5R8ExcurBfm/view?usp=share_link

## Acknowledgment
The structure of this repo is largely based on [sngan.pytorch](https://github.com/GongXinyuu/sngan.pytorch).