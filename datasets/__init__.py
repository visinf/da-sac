"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

from torch.utils import data
from tools.category import labels as SegLabels
from collections import namedtuple

from .dataloader_seg import DLSeg
from .dataloader_target import DataTarget

def get_class_names(args):
    return [l.name for l in SegLabels if l.name != "unlabeled"]

def get_num_classes(args):
    class_names = get_class_names(args)
    return len(class_names) # dont count ambiguous

def get_val_sets(train_split):
    if train_split == "train_game_9K":
        return ("val_game_1K", "train_cityscapes", "val_cityscapes", "val2_cityscapes"), "val2_cityscapes"
    elif train_split == "train_synthia_9K":
        return ("val_synthia", "train_cityscapes", "val_cityscapes", "val2_cityscapes"), "val2_cityscapes"
    else:
        raise NotImplementedError("Train split '{}' not recognised.".format(train_split))

def get_dataloader(args, cfg, split, num_classes, ngpus_per_node):
    assert split in ("train", "val")

    task = cfg.TRAIN.TASK
    assert task in ('train_game_9K', "train_synthia_9K"), "Unknown task: {}".format(task)

    # total batch size: # of GPUs * batch size per GPU
    batch_size = int(cfg.TRAIN.BATCH_SIZE / ngpus_per_node)
    workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    print("Dataloader: # workers {} / split {} ".format(workers, split))
    kwargs = {'num_workers': workers, 'pin_memory': True} if args.cuda else {}

    def _dataloader(dataset, batch_size, sampler=None, drop_last=False):
        sampler = data.distributed.DistributedSampler(dataset)
        return data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, \
                sampler=sampler, drop_last=drop_last, **kwargs), sampler

    if split == "train":
        # initialising source and target dataloaders
        dataset = DLSeg(cfg, task, renorm=cfg.DATASET.RENORM_SOURCE)
        if cfg.MODEL.BASELINE:
            dataset_target = DLSeg(cfg, cfg.TRAIN.TARGET)
        else:
            dataset_target = DataTarget(cfg, cfg.TRAIN.TARGET, num_classes, weights=cfg.DATASET.SAMPLE_WEIGHTS)

        dataset_target.set_num_samples(len(dataset))

        #assert cfg.TRAIN.NUM_GROUPS % ngpus_per_node == 0, \
        #        "Target batch not divisible: {} / {}".format(cfg.TRAIN.NUM_GROUPS, ngpus_per_node)

        # if we have more GPUs than the number of target images
        # load at least one target image;
        # we will handle the proper batch size before the forward pass
        batch_target = max(1, cfg.TRAIN.NUM_GROUPS // ngpus_per_node)
        print("Batch size SOURCE: ", batch_size)
        print("Batch size TARGET: ", batch_target, " / ", batch_target * cfg.TRAIN.GROUP_SIZE)

        return _dataloader(dataset, batch_size, drop_last=True), \
                _dataloader(dataset_target, batch_target, drop_last=True)
    else:
        val_sets, _ = get_val_sets(task)
        loaders = {}
        for val_set in val_sets:
            dataset = DLSeg(cfg, val_set)
            loader, sampler = _dataloader(dataset, batch_size)
            loaders[val_set] = loader

        return loaders
