# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import yaml
import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
from packaging import version

from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.NUM_EPOCHS = 200
# specifies the source data
__C.TRAIN.TASK = "train_game_9K"
# specifies the target data
__C.TRAIN.TARGET = "train_cityscapes"
# do not use source data if True
__C.TRAIN.TARGET_ONLY = False

# the batch size for the target is specified by
# NUM_GROUPS and GROUP_SIZE
# NUM_GROUPS is the number of image groups
# e.g. 2 with group size 4 means 
#   - two unique images
#   - four image versions for each (flips, crops, etc.)
#   - total batch size 8
__C.TRAIN.NUM_GROUPS = 4
__C.TRAIN.GROUP_SIZE = 2

# ignore certain class indicies when reporting val IoU
# (e.g. Synthia doesnt have all Cityscapes categories)
__C.VAL = AttrDict()
__C.VAL.IGNORE_CLASS = []

# ---------------------------------------------------------------------------- #
# Dataset options (+ augmentation options)
# ---------------------------------------------------------------------------- #
__C.DATASET = AttrDict()

__C.DATASET.CROP_SIZE = [512, 512]
__C.DATASET.VAL_CROP = True # use center crop for validation (rescale otherwise)
__C.DATASET.RND_CROP  = True
__C.DATASET.RND_BLUR = True
__C.DATASET.RND_GREYSCALE = 0.0
__C.DATASET.RND_HFLIP = True
__C.DATASET.RND_JITTER = 0.0
# scale range for target consistency
__C.DATASET.RND_ZOOM = [0.5, 1.2]
# horisontal flip with consistency
__C.DATASET.GUIDED_HFLIP = False

# source-specific augmentations
__C.DATASET.SRC_RND_BLUR = False
__C.DATASET.SRC_RND_JITTER = 0.4

# scale range for the source image
__C.DATASET.SCALE_FROM = 0.5
__C.DATASET.SCALE_TO = 1.5
# initial scale range for the target images
__C.DATASET.TARGET_SCALE = [1., 1.1]
__C.DATASET.ROOT = "data/datasets"

# path to the weights for importance sampling
__C.DATASET.SAMPLE_WEIGHTS = ""
# smoothing coefficient:
# 1 - uniform distribution
# 0 - fully determined by importance weights
__C.DATASET.SAMPLE_UNIFORM_PRIOR = 0.25

# See https://pytorch.org/docs/stable/torchvision/models.html
__C.DATASET.MEAN = [0.485, 0.456, 0.406]
__C.DATASET.STD = [0.229, 0.224, 0.225]

# renormalising source data with 
# target statistics
# TODO: this may not have any effect
# but maintained for reproducibility
__C.DATASET.RENORM_SOURCE = True

# GTA
__C.DATASET.SOURCE_MEAN = [0.481, 0.479, 0.465]
__C.DATASET.SOURCE_STD = [0.243, 0.239, 0.237]

# SYNTHIA
#__C.DATASET.SOURCE_MEAN = [0.315, 0.278, 0.248]
#__C.DATASET.SOURCE_STD = [0.225, 0.214, 0.210]

# Cityscapes stats
__C.DATASET.TARGET_MEAN = [0.300, 0.344, 0.297]
__C.DATASET.TARGET_STD = [0.175, 0.180, 0.177]

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()
__C.MODEL.ARCH = 'resnet38_fcn'
__C.MODEL.INIT_MODEL = ''
__C.MODEL.BASELINE = False
__C.MODEL.LR_TARGET = 1.0 # multiplier for the target loss
__C.MODEL.OPT = 'SGD'
__C.MODEL.OPT_NESTEROV = False
__C.MODEL.LR = 3e-4
__C.MODEL.BETA1 = 0.5
__C.MODEL.MOMENTUM = 0.9
__C.MODEL.WEIGHT_DECAY = 1e-5
# momentum for the moving class prior
# this is \gamma_\chi in the paper
__C.MODEL.STAT_MOMENTUM = 0.99
# momentum for the teacher/momentum net
# this is \gamma_\psi in the paper
__C.MODEL.NET_MOMENTUM = 0.99
# iteration interval for updating the momentum net
__C.MODEL.NET_MOMENTUM_ITER = 100
__C.MODEL.CONF_DISCOUNT = True
__C.MODEL.CONF_POOL_ON = True
__C.MODEL.CONF_POOL = "avg_pool"
__C.MODEL.FOCAL_P = 3
__C.MODEL.LOSS = "focal_ce_conf"
__C.MODEL.RUN_CONF_MOMENT = 0.9
__C.MODEL.RUN_CONF_UPPER = 0.75
__C.MODEL.RUN_CONF_LOWER = 0.2
# this is \beta hyperparameter in the paper
# the class prior gets divided by it
__C.MODEL.THRESHOLD_BETA = 1e-3

# ---------------------------------------------------------------------------- #
# Options for refinement
# ---------------------------------------------------------------------------- #
__C.LOG= AttrDict()
__C.LOG.ITER_VAL = 2
__C.LOG.ITER_TRAIN = 10
__C.LOG.ITER_TARGET = 4

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.TB = AttrDict()
__C.TB.IM_SIZE = (256, 256) # image resolution

# [Infered value]
__C.PYTORCH_VERSION_LESS_THAN_040 = False

def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value

cfg_from_list = merge_cfg_from_list


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
