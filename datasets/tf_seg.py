"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import random
import math
import numpy as np
import numbers
import collections
import torch
from PIL import Image, ImageFilter

import torchvision.transforms as tf
import torchvision.transforms.functional as F


class Compose:
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, *result):

        # allow for intermediate representations
        for t in self.segtransform:
            result = t(*result)

        return result


class ToTensorMask:

    def __toByteTensor(self, pic):
        return torch.from_numpy(np.array(pic, np.int32, copy=False))

    def __call__(self, image, labels, mask):
        image = F.to_tensor(image)
        labels = self.__toByteTensor(labels)
        mask = self.__toByteTensor(mask)

        return image, labels, mask

class CreateMask:

    def __call__(self, image, labels):
        
        # create mask as labels
        mask = Image.new("L", labels.size)

        return image, labels, mask

class Normalize:

    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, labels, mask):

        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)

        return image, labels, mask

class ApplyMask:

    def __init__(self, ignore_label):
        self.ignore_label = ignore_label

    def __call__(self, image, labels, mask):

        mask = mask > 0.

        image *= (1. - mask.type_as(image))
        labels[mask] = self.ignore_label

        return image, labels.squeeze(0).long()

class MaskRandAffine(object):

    def __init__(self, degree, scale):
        self.degrees = (-degree, degree)
        self.scale = scale
        self.translate = None
        self.shear = None

    def __call__(self, image, labels, mask):

        # getting the parameters
        ret = tf.RandomAffine.get_params(self.degrees, \
                                         self.translate, \
                                         self.scale, \
                                         self.shear, \
                                         image.size)

        # fill values will be replaced later
        image = F.affine(image, *ret, resample=Image.BILINEAR, fillcolor=0)
        labels = F.affine(labels, *ret, resample=Image.NEAREST, fillcolor=0)

        # keep track of the values to ignore later
        mask = F.affine(mask, *ret, resample=Image.NEAREST, fillcolor=1)

        return image, labels, mask

class MaskScale(object):

    def __init__(self, scale_to):
        self.scaled_size = scale_to[::-1] # width, height

    def __call__(self, image, labels, mask):

        image = image.resize(self.scaled_size, Image.BILINEAR)
        labels = labels.resize(self.scaled_size, Image.NEAREST)
        mask = mask.resize(self.scaled_size, Image.NEAREST)

        return image, labels, mask

class MaskRandScale(object):

    def __init__(self, scale_from, scale_to):
        self.scale = (scale_from, scale_to)
        self.aspect_ratio = None

    def __call__(self, image, labels, mask):

        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()

        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)

        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        scaled_size = (int(mask.size[0] * scale_factor_y), \
                       int(mask.size[1] * scale_factor_x))

        image = image.resize(scaled_size, Image.BILINEAR)
        labels = labels.resize(scaled_size, Image.NEAREST)
        mask = mask.resize(scaled_size, Image.NEAREST)

        return image, labels, mask

class MaskRandCrop:

    def __init__(self, size, pad_if_needed=False):
        self.size = size # (h, w)
        self.pad_if_needed = pad_if_needed

    def __pad(self, img, padding_mode='constant', fill=0):

        # pad the width if needed
        pad_width = self.size[1] - img.size[0]
        pad_height = self.size[0] - img.size[1]
        if self.pad_if_needed and (pad_width > 0 or pad_height > 0):
            pad_l = max(0, pad_width // 2)
            pad_r = max(0, pad_width - pad_l)
            pad_t = max(0, pad_height // 2)
            pad_b = max(0, pad_height - pad_t)
            img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), fill, padding_mode)

        return img

    def __call__(self, image, labels, mask):

        image = self.__pad(image)
        labels = self.__pad(labels)
        mask = self.__pad(mask, fill=1)

        i, j, h, w = tf.RandomCrop.get_params(image, self.size)

        image = F.crop(image, i, j, h, w)
        labels = F.crop(labels, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return image, labels, mask

class MaskCenterCrop:

    def __init__(self, size):
        self.size = size

    def __call__(self, image, labels, mask):

        image = F.center_crop(image, self.size)
        labels = F.center_crop(labels, self.size)
        mask = F.center_crop(mask, self.size)

        return image, labels, mask

class MaskRandHFlip:

    def __call__(self, image, labels, mask):

        if random.random() > 0.5:
            image = F.hflip(image)
            labels = F.hflip(labels)
            mask = F.hflip(mask)

        return image, labels, mask

class RandGaussianBlur:

    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, radius=1.0):
        self.radius = radius

    def __call__(self, image, labels, mask):

        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(self.radius))

        return image, labels, mask

class MaskRandJitter:

    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, jitter, p=0.5):
        self.jitter = tf.ColorJitter(brightness=jitter, \
                                     contrast=jitter, \
                                     saturation=jitter, \
                                     hue=min(0.5, jitter))

    def __call__(self, image, labels, mask):

        if random.random() < 0.5:
            image = self.jitter(image)

        return image, labels, mask
