# EnvMat

## Overview
EnvMat is a deep learning model designed to simultaneously generate SVBRDF maps and Environment maps. Recent research on PBR map generation has predominantly focused on creating only SVBRDF maps. 

As a result, there is a limitation in producing generalized outcomes across various environments. To address this issue, our study aims to generate both SVBRDF maps and Environment maps concurrently.


## Repository details
This repo is a slightly modified version of MatFuse (https://github.com/giuvecchio/matfuse-sd), which is based on the original latent diffusion (https://github.com/CompVis/stable-diffusion).

For details on installation, inference, and training, please refer to the MatFuse repository mentioned above, as they are the same.


## Prepare Data
The SVBRDF maps data used to train EnvMat can be found at MatSynth (https://huggingface.co/datasets/gvecchio/MatSynth). Additionally, another Environmental map dataset used for training is the Laval Indoor Dataset (http://hdrdb.com/indoor/).

The preprocessing scripts required for training data are located in the `data_scripts` folder. First, perform augmentation using `make_crops.py`, then render the augmented maps with `rendering.py` to prepare them for training.
