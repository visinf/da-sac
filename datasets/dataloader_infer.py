"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0

Description: Data loader at test time (single scale)
"""

import os
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transform
import torchvision.transforms.functional as tf

from .dataloader_base import DLBase

def get_dataloader(loader_name, *args, **kwargs):

    if loader_name == "cityscapes":
        dataloader = DLCityscapesInfer
    else:
        dataloader = DLInfer

    print("Dataloader: {}".format(dataloader.__name__))
    return dataloader(*args, **kwargs)

class DLInfer(DLBase):

    def __init__(self, cfg, split, ignore_labels=[], root=os.path.expanduser('./data')):
        super(DLInfer, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.ignore_labels = ignore_labels

        # train/val/test splits are pre-cut
        split_fn = os.path.join(self.root, self.split + ".txt")
        assert os.path.isfile(split_fn)

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
            
                if not _mask is None:
                    _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                    assert os.path.isfile(_mask), '%s not found' % _mask
                self.masks.append(_mask)

    def __len__(self):
        return len(self.images)

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

    def transform(self, image, mask):

        image = tf.to_tensor(image)
        imnorm = transform.Normalize(self.MEAN, self.STD)
        image = imnorm(image)

        mask = torch.from_numpy(np.array(mask))
        mask = self.remove_labels(mask)

        return image, mask

    def extract_name(self, image_path):
        # getting the basename
        basename = os.path.basename(image_path)
        # removing the extension
        basename = os.path.splitext(basename)[0]
        return basename

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert('RGB')

        if self.masks[index] is None:
            mask = Image.new('L', image.size, color=255)
        else:
            mask = Image.open(self.masks[index])

        # general resize, normalize and toTensor
        image, mask = self.transform(image, mask)

        return image, mask, self.extract_name(self.images[index])

class DLCityscapesInfer(DLInfer):

    def extract_name(self, image_path):
        """Images have _leftImg8bit suffix, while masks
        have _gtFine or other suffix"""
        basename = super().extract_name(image_path)
        return basename.replace("_leftImg8bit", "_gtFine_labelIds")
