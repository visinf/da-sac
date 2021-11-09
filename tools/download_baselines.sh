"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0

Description:
Download all pre-trained models into the current directory.
Tip: Move this file to snapshots/cityscapes/baselines and execute
"""

#!/bin/bash

echo "Downloading baseline models"

ROOT_URL=download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines

download_model () {
  echo $1 $ROOT_URL
  curl $ROOT_URL/$1 --create-dirs -o $1 
}

# ResNet-101 / GTA
download_model resnet101_gta/baseline_abn_e040.pth
download_model resnet101_gta/final_e136.pth

# ResNet-101 / SYNTHIA
download_model resnet101_synthia/baseline_abn_e090.pth
download_model resnet101_synthia/final_e164.pth

# VGG-16 / GTA (DeepLabv2)
download_model vgg16_gta/baseline_abn_e115.pth
download_model vgg16_gta/final_e184.pth

# VGG-16 / SYNTHIA (DeepLabv2)
download_model vgg16_synthia/baseline_abn_e070.pth
download_model vgg16_synthia/final_e164.pth

# VGG-16 / GTA (FCN)
download_model vgg16fcn_gta/baseline_abn_e040.pth
download_model vgg16fcn_gta/final_e112.pth

# VGG-16 / SYNTHIA (FCN)
download_model vgg16fcn_synthia/baseline_abn_e040.pth
download_model vgg16fcn_synthia/final_e098.pth
