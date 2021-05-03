"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import re
import random
import numpy as np
import time
import sys
import math

from PIL import Image, ImagePalette, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from functools import partial, reduce

from .dataloader_base import DLBase
import datasets.tf_seg as tf


class DLSeg(DLBase):

    def __init__(self, cfg, split, ignore_labels=[], \
                 root=os.path.expanduser('./data'), renorm=False):

        super(DLSeg, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.ignore_labels = ignore_labels

        # train/val/test splits are pre-cut
        split_fn = os.path.join(self.root, self.split + ".txt")
        assert os.path.isfile(split_fn), "File {} does not exist".format(split_fn)

        self.images = []
        self.masks = []

        with open(split_fn, "r") as lines:
            for line in lines:
                _image = line.strip("\n").split(' ')

                _mask = None
                if len(_image) == 2:
                    _image, _mask = _image
                else:
                    assert len(_image) == 1
                    _image = _image[0]

                _image = os.path.join(cfg.DATASET.ROOT, _image.lstrip('/'))
                assert os.path.isfile(_image), '%s not found' % _image
                self.images.append(_image)

                if _mask is None:
                    self.masks.append(None)
                else:
                    _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                    assert os.path.isfile(_mask), '%s not found' % _mask
                    self.masks.append(_mask)

        # definint data augmentation:
        tfs = [tf.CreateMask()]
        print("Dataloader: {}".format(split), " #", len(self.images))
        if split.startswith("train"):
            print("\t {}: w/ data augmentation".format(split))
            tfs.append(tf.MaskRandScale(cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO))

            if cfg.DATASET.SRC_RND_BLUR:
                tfs.append(tf.RandGaussianBlur())

            if cfg.DATASET.RND_HFLIP:
                tfs.append(tf.MaskRandHFlip())

            if cfg.DATASET.SRC_RND_JITTER > 0:
                tfs.append(tf.MaskRandJitter(cfg.DATASET.RND_JITTER))

            if cfg.DATASET.RND_CROP:
                tfs.append(tf.MaskRandCrop(cfg.DATASET.CROP_SIZE, pad_if_needed=True))
        else:
            print("\t {}: no augmentation".format(split))
            if cfg.DATASET.VAL_CROP:
                tfs.append(tf.MaskCenterCrop(cfg.DATASET.CROP_SIZE))
            else:
                tfs.append(tf.MaskScale(cfg.DATASET.CROP_SIZE))

        tfs_post = [tf.ToTensorMask()]
        if renorm:
            # updating the mean and std
            mean = np.array(self.MEAN)
            std = np.array(self.STD)
            mean_src = np.array(cfg.DATASET.SOURCE_MEAN)
            mean_tgt = np.array(cfg.DATASET.TARGET_MEAN)
            stdv_src = np.array(cfg.DATASET.SOURCE_STD)
            stdv_tgt = np.array(cfg.DATASET.TARGET_STD)
            mean = tuple(mean_src - stdv_src / stdv_tgt * (mean_tgt - mean))
            std = tuple(stdv_src * std / stdv_tgt)
            tfs_post += [tf.Normalize(mean=mean, std=std)]
        else:
            tfs_post += [tf.Normalize(mean=self.MEAN, std=self.STD)]

        tfs_post += [tf.ApplyMask(255)]

        # final transformation
        self.tf_augm = tf.Compose(tfs)
        self.tf_post = tf.Compose(tfs_post)

        self._num_samples = len(self.images)

    def set_num_samples(self, n):
        print("Re-setting # of samples: {:d} -> {:d}".format(self._num_samples, n))
        self._num_samples = n

    def __len__(self):
        return self._num_samples

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image

    def __getitem__(self, index):

        index = index % len(self.images)
        image = Image.open(self.images[index]).convert('RGB')

        if self.masks[index] is None:
            mask = Image.new('L', image.size)
        else:
            mask = Image.open(self.masks[index]).convert('L')

        if "game" in self.split:
            image = image.resize((1914, 1052), Image.BILINEAR)
            mask = mask.resize((1914, 1052), Image.NEAREST)

        assert (image.size[0] == mask.size[0] and image.size[1] == mask.size[1]), \
            "Image & mask shape mismatch: {} != {}".format(self.images[index], self.masks[index])

        augms = self.tf_augm(image, mask)
        image, mask = self.tf_post(*augms)

        return image, mask
