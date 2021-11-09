#!/bin/bash

echo "Downloading importance sampling weights"

ROOT_URL=download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/cs_weights

download_model () {
  echo $1 $ROOT_URL
  curl $ROOT_URL/$1 --create-dirs -o $1 
}

# ResNet-101 / GTA
download_model cs_weights_resnet101_gta.data

# ResNet-101 / SYNTHIA
download_model cs_weights_resnet101_synthia.data

# VGG-16 / GTA (DeepLabv2)
download_model cs_weights_vgg16_gta.data

# VGG-16 / SYNTHIA (DeepLabv2)
download_model cs_weights_vgg16_synthia.data

# VGG-16 / GTA (FCN)
download_model cs_weights_vgg16fcn_gta.data

# VGG-16 / SYNTHIA (FCN)
download_model cs_weights_vgg16fcn_synthia.data
