# Self-supervised Augmentation Consistency <br/> for Adapting Semantic Segmentation
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" height="20" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

This repository contains the official implementation of our paper:

**Self-supervised Augmentation Consistency for Adapting Semantic Segmentation**<br>
[Nikita Araslanov](https://arnike.github.io) and [Stefan Roth](https://www.visinf.tu-darmstadt.de/visinf/team_members/sroth/sroth.en.jsp)<br>
CVPR 2021. [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Araslanov_Self-Supervised_Augmentation_Consistency_for_Adapting_Semantic_Segmentation_CVPR_2021_paper.pdf)] [[supp](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Araslanov_Self-Supervised_Augmentation_Consistency_CVPR_2021_supplemental.pdf)] [[arXiv](https://arxiv.org/abs/2105.00097)]

| <img src="assets/stuttgart.gif" alt="drawing" width="420"/><br> |
|:--:|
| <p align="left">We obtain state-of-the-art accuracy of adapting semantic <br> segmentation by enforcing consistency across photometric <br> and similarity transformations. We use neither style transfer <br> nor adversarial training.</p> |


Contact: Nikita Araslanov *fname.lname* (at) visinf.tu-darmstadt.de


---

## Installation
**Requirements.** To reproduce our results, we recommend Python >=3.6, PyTorch >=1.4, CUDA >=10.0. At least two Titan X GPUs (12Gb) or equivalent are required for VGG-16; ResNet-101 and VGG-16/FCN need four.

1. create conda environment:
```
conda create --name da-sac
source activate da-sac
```

2. install PyTorch >=1.4 (see [PyTorch instructions](https://pytorch.org/get-started/locally/)). For example,

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

3. install the dependencies:
```
pip install -r requirements.txt
```

4. download data ([Cityscapes](https://www.cityscapes-dataset.com/downloads/), [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), [SYNTHIA](https://synthia-dataset.net/downloads/)) and create symlinks in the ```./data``` folder, as follows:

```
./data/cityscapes -> <symlink to Cityscapes>
./data/cityscapes/gtFine2/
./data/cityscapes/leftImg8bit/

./data/game -> <symlink to GTA>
./data/game/labels_cs
./data/game/images

./data/synthia  -> <symlink to SYNTHIA>
./data/synthia/labels_cs
./data/synthia/RGB
```
Note that all ground-truth label IDs (Cityscapes, GTA5 and SYNTHIA) should be converted to Cityscapes [train IDs](assets/train_IDs.md).
The label directories in the above example (```gtFine2```, ```labels_cs```) therefore refer not to the original labels, but to these converted semantic maps.

## Training
Training from ImageNet initialisation proceeds in three steps:
1. Training the baseline (ABN)
2. Generating the weights for importance sampling
3. Training with augmentation consistency from the ABN baseline

### 1. Training the baseline (ABN)
Here the input are ImageNet models available from the official PyTorch repository. We provide the links to those models for convenience.
| Backbone | Link |
|---|---|
| ResNet-101 | [resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) (171M) |
| VGG-16 | [vgg16_bn-6c64b313.pth](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth) (528M) |

By default, these models should be placed in ./models/pretrained/ (though configurable with ```MODEL.INIT_MODEL```).

To run the training
```
bash ./launch/train.sh [gta|synthia] [resnet101|vgg16|vgg16fcn] base
```
where the first argument specifies the source domain, the second determines the network architecture.
The third argument ```base``` instructs to run the training of the baseline.

If you would like to skip this step, you can use our pre-trained models:

**Source domain: GTA5**
| Backbone | Arch. | IoU (val) | Link | MD5 |
|---|---|:-:|---|---|
| ResNet-101 | DeepLabv2 | 40.8 | [baseline_abn_e040.pth (336M)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/resnet101_gta/baseline_abn_e040.pth) | `9fe17[...]c11fc` |
| VGG-16 | DeepLabv2 | 37.1 | [baseline_abn_e115.pth (226M)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/vgg16_gta/baseline_abn_e115.pth) | `d4ffc[...]ef755` |
| VGG-16 | FCN | 36.7 | [baseline_abn_e040.pth (1.1G)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/vgg16fcn_gta/baseline_abn_e040.pth) | `aa2e9[...]bae53` |


**Source domain: SYNTHIA**
| Backbone | Arch. | IoU (val) | Link | MD5 |
|---|---|:-:|---|---|
| ResNet-101 | DeepLabv2 | 36.3 | [baseline_abn_e090.pth (336M)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/resnet101_synthia/baseline_abn_e090.pth) | `b3431[...]d1a83` |
| VGG-16 | DeepLabv2 | 34.4 | [baseline_abn_e070.pth (226M)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/vgg16_synthia/baseline_abn_e070.pth) | `3af24[...]5b24e` |
| VGG-16 | FCN | 31.6 | [baseline_abn_e040.pth (1.1G)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/vgg16fcn_synthia/baseline_abn_e040.pth) | `5f457[...]e4b3a` |

**Tip:** You can download these files (as well as the final models below) with ```tools/download_baselines.sh```:
```bash
cp tools/download_baselines.sh snapshots/cityscapes/baselines/
cd snapshots/cityscapes/baselines/
bash ./download_baselines.sh
```

### 2. Generating weights for importance sampling 
To generate the weights you need to
1. generate mask predictions with your baseline (see [inference](#inference) below);
2. run ```tools/compute_image_weights.py``` that reads in those predictions and counts the predictions per each class.

If you would like to skip this step, you can use our weights we computed for the ABN baselines above:
| Backbone | Arch. | Source: GTA5 | Source: SYNTHIA |
|---|---|---|---|
| ResNet-101 | DeepLabv2 | [cs_weights_resnet101_gta.data](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/cs_weights/cs_weights_resnet101_gta.data) | [cs_weights_resnet101_synthia.data](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/cs_weights/cs_weights_resnet101_synthia.data) |
| VGG-16 | DeepLabv2 | [cs_weights_vgg16_gta.data](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/cs_weights/cs_weights_vgg16_gta.data) | [cs_weights_vgg16_synthia.data](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/cs_weights/cs_weights_vgg16_synthia.data) |
| VGG-16 | FCN | [cs_weights_vgg16fcn_gta.data](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/cs_weights/cs_weights_vgg16fcn_gta.data) | [cs_weights_vgg16fcn_synthia.data](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/cs_weights/cs_weights_vgg16fcn_synthia.data) |

**Tip:** The bash script ```data/download_weights.sh``` will download all these importance sampling weights in the current directory.

### 3. Training with augmentation consistency
To train the model with augmentation consistency, we use the same shell script as in step 1, but without the argument ```base```:
```
bash ./launch/train.sh [gta|synthia] [resnet101|vgg16|vgg16fcn]
```
Make sure to specify your baseline snapshot with ```RESUME``` bash variable set in the environment (```export RESUME=...```) or directly in the shell script (commented out by default).

We provide our final models for download.

**Source domain: GTA5**
| Backbone | Arch. | IoU (val) |  IoU (test) | Link | MD5 |
|---|---|:-:|:-:|---|---|
| ResNet-101 | DeepLabv2 | 53.8 | 55.7 | [final_e136.pth (504M)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/resnet101_gta/final_e136.pth) | `59c16[...]5a32f` |
| VGG-16 | DeepLabv2 | 49.8 | 51.0 | [final_e184.pth (339M)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/vgg16_gta/final_e184.pth) | `0accb[...]d5881` |
| VGG-16 | FCN | 49.9 | 50.4 | [final_e112.pth (1.6G)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/vgg16fcn_gta/final_e112.pth) | `e69f8[...]f729b` |

**Source domain: SYNTHIA**
| Backbone | Arch. | IoU (val) |  IoU (test) | Link | MD5 |
|---|---|:-:|:-:|---|---|
| ResNet-101 | DeepLabv2 | 52.6 | 52.7 | [final_e164.pth (504M)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/resnet101_synthia/final_e164.pth) | `a7682[...]db742` |
| VGG-16 | DeepLabv2 | 49.1 | 48.3 | [final_e164.pth (339M)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/vgg16_synthia/final_e164.pth) | `c5b31[...]5fdb7` |
| VGG-16 | FCN | 46.8 | 45.8 | [final_e098.pth (1.6G)](https://download.visinf.tu-darmstadt.de/data/2021-cvpr-araslanov-da-sac/snapshots/baselines/vgg16fcn_synthia/final_e098.pth) | `efb74[...]845cc` |

## Inference and evaluation

### Inference
To run single-scale inference from your snapshot, use ```infer_val.py```.
The bash script ```launch/infer_val.sh``` provides an easy way to run the inference by specifying a few variables:
```
# validation/training set
FILELIST=[val_cityscapes|train_cityscapes] 
# configuration used for training
CONFIG=configs/[deeplabv2_vgg16|deeplab_resnet101|fcn_vgg16]_train.yaml
# the following 3 variables effectively specify the path to the snapshot
EXP=...
RUN_ID=...
SNAPSHOT=...
# the snapshot path is defined as
# SNAPSHOT_PATH=snapshots/cityscapes/${EXP}/${RUN_ID}/${SNAPSHOT}.pth
```

### Evaluation
Please use the Cityscapes' official evaluation tool ```evalPixelLevelSemanticLabeling``` from [Cityscapes scripts](https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts) for evaluating your results.

## Citation
We hope you find our work useful. If you would like to acknowledge it in your project, please use the following citation:
```
@inproceedings{Araslanov:2021:DASAC,
    author    = {Araslanov, Nikita and Roth, Stefan},
    title     = {Self-Supervised Augmentation Consistency for Adapting Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {15384-15394}
}
```
