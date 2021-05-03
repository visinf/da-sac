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
import copy
import bisect

from PIL import Image, ImagePalette, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from functools import partial, reduce

from .dataloader_base import DLBase
import datasets.tf_target as tf


class DataTarget(DLBase):

    IGNORE_LABEL = 255

    def __init__(self, cfg, split, num_classes, weights="", ignore_labels=[], root=os.path.expanduser('./data')):
        super(DataTarget, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.ignore_labels = ignore_labels
        self.num_classes = num_classes

        # train/val/test splits are pre-cut
        split_fn = os.path.join(self.root, self.split + ".txt")
        assert os.path.isfile(split_fn)

        self.images = []
        self.masks = []

        # index -> name, name -> index
        self.sample_index = {}

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

                # TODO: Do we need to handle this case?
                if _mask is None:
                    self.masks.append(None)
                else:
                    _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                    assert os.path.isfile(_mask), '%s not found' % _mask
                    self.masks.append(_mask)

                    mindex, mname = len(self.masks) - 1, os.path.basename(_mask)
                    self.sample_index[mname] = mindex

        self._num_samples = len(self.images)

        # Init sampling
        sample_weights = None
        if len(weights):
            if os.path.isfile(weights):
                print("Loading sample weights: {}".format(weights))
                sample_weights = torch.load(weights)
            else:
                print("Path to sample weights NOT found: {}".format(weights))

        # contains list of indices and associated weights
        self.sample_groups = self.init_sampling(self._num_samples, sample_weights, \
                                                prior_weight=cfg.DATASET.SAMPLE_UNIFORM_PRIOR)

        # definint data augmentation:
        tfs = [tf.CreateMask()]
        tfs_augm = []

        print("Dataloader: {}".format(split), " #", len(self.images))
        if not split.startswith("val"):
            print("\t {}: w/ data augmentation".format(split))

            tfs.append(tf.MaskScale(cfg.DATASET.CROP_SIZE))
            tfs.append(tf.MaskRandScale(cfg.DATASET.TARGET_SCALE[0], cfg.DATASET.TARGET_SCALE[1]))
            tfs.append(tf.MaskRandCrop(cfg.DATASET.CROP_SIZE, pad_if_needed=True))

            if cfg.DATASET.RND_HFLIP:
                tfs.append(tf.MaskRandHFlip())

            if cfg.DATASET.GUIDED_HFLIP:
                tfs.append(tf.GuidedRandHFlip())

            # this will add affine transformation
            if cfg.DATASET.RND_ZOOM[1] - cfg.DATASET.RND_ZOOM[0] > 0:
                tfs.append(tf.MaskRandScaleCrop(cfg.DATASET.RND_ZOOM))

            if cfg.DATASET.RND_BLUR:
                tfs_augm.append(tf.RandGaussianBlur())

            if cfg.DATASET.RND_JITTER > 0:
                tfs_augm.append(tf.MaskRandJitter(cfg.DATASET.RND_JITTER))

            if cfg.DATASET.RND_GREYSCALE > 0:
                tfs_augm.append(tf.MaskRandGreyscale(cfg.DATASET.RND_GREYSCALE))
        else:
            print("\t {}: no augmentation".format(split))
            if cfg.DATASET.VAL_CROP:
                tfs.append(tf.MaskCenterCrop(cfg.DATASET.CROP_SIZE))
            else:
                tfs.append(tf.MaskScale(cfg.DATASET.CROP_SIZE))

        self.tf_pre = tf.Compose(tfs)
        self.tf_augm = tf.Compose(tfs_augm)

        tfs_post = [tf.ToTensorMask(),
                    tf.Normalize(mean=self.MEAN, std=self.STD),
                    tf.ApplyMask(-1)]

        # post-transformations
        self.tf_post = tf.Compose(tfs_post)
        self._num_samples = len(self.images)

    def set_num_samples(self, n):
        print("Re-setting # of samples: {:d} -> {:d}".format(self._num_samples, n))
        self._num_samples = n

    def init_sampling(self, num_samples, loaded_weights = None, prior_weight = 0.7):
        """Weights are given as
        basename -> [pixel_fraction_class_1, pixel_fraction_class_2, ...]
        """

        # uniform prior
        prior = 1. / num_samples

        # uniform sampling
        sample_groups = {}

        if not loaded_weights is None:
            weights = [prior_weight * prior for _ in range(num_samples)]
            weights_uniform = [prior for _ in range(num_samples)]

            for cid in range(self.num_classes):
                #sample_groups[cid] = weights.copy()
                if cid in self.cfg.VAL.IGNORE_CLASS:
                    sample_groups[cid] = weights_uniform.copy() # copy
                else:
                    sample_groups[cid] = weights.copy() # copy

            assert len(loaded_weights) == num_samples, \
                    "Loaded weights {} do not match # of loaded images {}".format(num_samples, len(loaded_weights))
            # iterate through images
            for name, stat in loaded_weights.items():
                sample_id = self.sample_index[name]
                for cid, val in stat.items():
                    sample_groups[cid][sample_id] += (1. - prior_weight) * val

            # for ignored classes use uniform sampling
            weights = [prior for _ in range(num_samples)]
            for cid in self.cfg.VAL.IGNORE_CLASS:
                sample_groups[cid] = weights.copy()
        else:
            # uniform sampling
            weights = [prior for _ in range(num_samples)]

            for cid in range(self.num_classes):
                sample_groups[cid] = weights.copy() # copy

        # accumulating
        for cid in range(self.num_classes):

            sample_group = sample_groups[cid]
            for sample_id in range(1, len(sample_group)):
                sample_group[sample_id] += sample_group[sample_id - 1]

            assert abs(sample_group[-1] - 1.) < 1e-3, "Cumulative weights [{}] do not add up {}".format(cid, sample_group[-1])

        #torch.save(sample_groups, "sample_groups.pth")

        return sample_groups

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

    def _get_affine(self, params):
        # construct affine operator
        affine = torch.zeros(self.cfg.TRAIN.GROUP_SIZE, 2, 3)

        aspect_ratio = float(self.cfg.DATASET.CROP_SIZE[0]) / \
                            float(self.cfg.DATASET.CROP_SIZE[1])
        for i, (dy,dx,alpha,scale,flip) in enumerate(params):
            # R inverse
            sin = math.sin(alpha * math.pi / 180.)
            cos = math.cos(alpha * math.pi / 180.)

            # inverse, note how flipping is incorporated
            affine[i,0,0], affine[i,0,1] = flip * cos, sin * aspect_ratio
            affine[i,1,0], affine[i,1,1] = -sin / aspect_ratio, cos

            # T inverse Rinv * t == R^T * t
            affine[i,0,2] = -1. * (cos * dx + sin * dy)
            affine[i,1,2] = -1. * (-sin * dx + cos * dy)

            # T
            affine[i,0,2] /= float(self.cfg.DATASET.CROP_SIZE[1] // 2)
            affine[i,1,2] /= float(self.cfg.DATASET.CROP_SIZE[0] // 2)

            # scaling
            affine[i] *= scale

        return affine

    def _get_affine_inv(self, affine, params):

        aspect_ratio = float(self.cfg.DATASET.CROP_SIZE[0]) / \
                            float(self.cfg.DATASET.CROP_SIZE[1])

        affine_inv = affine.clone()
        affine_inv[:,0,1] = affine[:,1,0] * aspect_ratio**2
        affine_inv[:,1,0] = affine[:,0,1] / aspect_ratio**2
        affine_inv[:,0,2] = -1 * (affine_inv[:,0,0] * affine[:,0,2] + affine_inv[:,0,1] * affine[:,1,2])
        affine_inv[:,1,2] = -1 * (affine_inv[:,1,0] * affine[:,0,2] + affine_inv[:,1,1] * affine[:,1,2])

        # scaling
        affine_inv /= torch.Tensor(params)[:, 3].view(-1,1,1)**2

        return affine_inv

    def __getitem__(self, index):

        # select category uniformly
        cat_idx = index % len(self.sample_groups)
        sample_weights = self.sample_groups[cat_idx]

        # select sample based on the weights
        rand_sample = random.uniform(0, sample_weights[-1])
        select_idx = bisect.bisect_left(sample_weights, rand_sample)

        image = Image.open(self.images[select_idx]).convert('RGB')
        if self.masks[select_idx] is None:
            mask = Image.new('L', image.size, (self.IGNORE_LABEL, ))
        else:
            mask = Image.open(self.masks[select_idx]).convert('L')

        assert (image.size[0] == mask.size[0] and image.size[1] == mask.size[1]), \
            "Image & mask shape mismatch: {} != {}".format(self.images[index], self.masks[index])

        # Copy N copies of an image
        images = [image.copy() for _ in range(self.cfg.TRAIN.GROUP_SIZE)]
        masks = [mask.copy() for _ in range(self.cfg.TRAIN.GROUP_SIZE)]

        augms = self.tf_pre(images, masks)

        affine_params = augms[-1]
        augms = augms[:-1]

        augms2 = copy.deepcopy(augms)
        augms1 = self.tf_augm(*augms)

        images1, masks = self.tf_post(*augms1)
        images2, _ = self.tf_post(*augms2)

        # concatenating along leading axis
        images1 = torch.cat([x.unsqueeze(0) for x in images1], 0)
        images2 = torch.cat([x.unsqueeze(0) for x in images2], 0)
        masks = torch.cat([x.unsqueeze(0) for x in masks], 0)

        affine = self._get_affine(affine_params)
        affine_inv = self._get_affine_inv(affine, affine_params)

        return images1, masks, images2, affine, affine_inv
