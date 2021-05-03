"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os

from .deeplabv2 import DeepLabV2_ResNet101, DeepLabV2_VGG16
from .fcn import VGG16_FCN8s
from functools import partial
from .sac import SAC, SAC_Baseline

def get_model(cfg, rank, *args, **kwargs):

    models = {
        'deeplabv2_resnet101': DeepLabV2_ResNet101,
        'deeplabv2_vgg16_bn': partial(DeepLabV2_VGG16, use_bn=True),
        'fcn_vgg16_bn': partial(VGG16_FCN8s, use_bn=True)
    }

    if len(cfg.INIT_MODEL) > 0 and os.path.isfile(cfg.INIT_MODEL):
        kwargs["pretrained"] = cfg.INIT_MODEL
    else:
        print("Backbone model not found: {}".format(cfg.INIT_MODEL))

    # freeze BN
    # when jointly training with the consistency loss
    kwargs["freeze_bn"] = not cfg.BASELINE

    # training BN only in pre-training (ABN)
    backbone = models[cfg.ARCH.lower()](*args, **kwargs)

    if cfg.BASELINE:
        return SAC_Baseline(cfg, backbone, rank, **kwargs)
    
    # creating a slow copy
    slow_copy = models[cfg.ARCH.lower()](*args, **kwargs)

    # final model
    return SAC(cfg, backbone, slow_copy, rank, **kwargs)
