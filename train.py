"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

from __future__ import print_function

import os
import copy
import sys
import numpy as np
import time
import math
import random
import builtins
import setproctitle

from functools import partial

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

from datasets import *
from models import get_model

from base_trainer import BaseTrainer

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from utils.timer import Timer
from utils.stat_manager import StatManager
from utils.metrics import Jaccard
from utils.sys_tools import find_free_port

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

DEBUG = False
EVALUATE = not DEBUG
EVALUATE_SOURCE = False
SNAPSHOT = not DEBUG

class Trainer(BaseTrainer):

    def __init__(self, args, cfg, gpu, ngpus_per_node):

        main_process = args.rank % ngpus_per_node == 0
        super(Trainer, self).__init__(args, cfg, main_process)

        self.gpu = gpu
        self.world_size = args.world_size
        self.device = torch.device("cuda", gpu)

        # TODO: remove
        self.nclass = get_num_classes(args)
        self.classNames = get_class_names(args)
        assert self.nclass == len(self.classNames)

        self.classIndex = {}
        for i, cname in enumerate(self.classNames):
            print("{:>5} -> {}".format(cname, i))
            self.classIndex[cname] = i

        # train loader for target domain
        train_source, train_target = get_dataloader(args, cfg, 'train', self.nclass, ngpus_per_node)
        self.loader_source, self.sampler_source = train_source
        self.loader_target, self.sampler_target = train_target

        # just an alias
        self.denorm = self.loader_source.dataset.denorm

        # val loaders for source and target domains
        self.valloaders = get_dataloader(args, cfg, 'val', self.nclass, ngpus_per_node)

        # writers (only main)
        val_sets, self.testset = get_val_sets(cfg.TRAIN.TASK)
        self.writer_val = {}
        for val_set in val_sets:
            logdir_val = os.path.join(args.logdir, val_set)
            self.writer_val[val_set] = SummaryWriter(logdir_val) if self.main_process else None

        # initialising the model
        self.net = get_model(cfg.MODEL, self.gpu, num_classes=self.nclass, \
                             criterion=nn.CrossEntropyLoss(ignore_index=255, reduction="none"))
        print(self.net)

        # optimizer using different LR
        net_params = self.net.parameter_groups(cfg.MODEL.LR, cfg.MODEL.WEIGHT_DECAY)
        self.optim = self.get_optim(net_params, cfg.MODEL)

        print("# of params: ", len(list(self.net.parameters())))

        self.fixed_batch = None

        # using cuda
        print(">>> Distributed training: ", args.rank, gpu)
        torch.cuda.set_device(gpu)
        self.net.cuda(gpu)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[gpu])

        # checkpoint management
        self.checkpoint.create_model(self.net, self.optim)
        if not args.resume is None:
            self.start_epoch, self.best_score = self.checkpoint.load(args.resume, self.device)

        print(self.optim)

        if self.cfg.MODEL.BASELINE:
            # note we don't backprop
            self.step_target = partial(self.step, train=False)
        else:
            self.step_target = partial(self._step_target, train=True, visualise=False)

    def step(self, epoch, batch_source, train=False, visualise=False, \
                    save_batch=False, writer=None, tag="train"):

        image, masks_gt = batch_source

        image = image.cuda(self.gpu, non_blocking=True)
        masks_gt = masks_gt.cuda(self.gpu, non_blocking=True)

        # source forward pass
        losses, logits = self.net(image, masks_gt)

        if train:
            # vanilla baseline mode: just minimise CE
            self.optim.zero_grad()
            losses["loss_ce"].mean().backward()
            if self.cfg.MODEL.BASELINE:
                # in baseline mode we update immediately,
                # otherwise accumulate source and target losses
                # before the backward pass
                self.optim.step()

        if visualise:
            self._visualise(epoch, image, masks_gt, logits, writer, tag)

        if save_batch:
            # Saving batch for visualisation
            self.save_fixed_batch(tag, batch_source)

        # summarising the losses
        # into python scalars
        losses_ret = {}
        for key, val in losses.items():
            losses_ret[key] = val.mean().item()

        # only for convenience to compute validation IoU
        logits["mask_gt"] = masks_gt
        return losses_ret, logits

    def _prep_batch(self, tensor):
        """Given a batch of size B split between the GPUs.
        Manual splitting is required, since for each image we 
        may extract more crops than can fit on one GPU.
        We split it here, but when calculating the consistency
        loss, we merge the output with all_gather.

        Args:
            tensor: a tensor of size [B,T,...]
        Returns:
            out: a sliced tensor of size [B1, ...]
        """

        # number of unique target images
        N = self.cfg.TRAIN.NUM_GROUPS
        # size of the batch per target image
        # (including original image)
        L = self.cfg.TRAIN.GROUP_SIZE

        # compute access index for this gpu
        assert (N * L) % self.world_size == 0, "Batch size does not fit world size"

        # number of images we will fit on each GPU
        batch_per_gpu = N * L // self.world_size

        # loading to GPU
        tensor = tensor.cuda(self.gpu)

        # if the whole set fits on 1 GPU, skip
        if batch_per_gpu >= L:
            return tensor.flatten(0, 1)           

        #B = tensor.size(0) # loaded batch size
        T = tensor.size(1) # loaded sequence size

        # sanity check
        assert T == L, "Loaded sequence is incorrect {} vs. {}".format(T, L)
        out_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(out_list, tensor)

        # if we were to view the batch as [B*T, -1]
        # this will be the select for this GPU
        batch_index = self.gpu * batch_per_gpu

        # but in our case, B*T is scattered in the list,
        # so we need to index the list elements first
        index0 = batch_index // T
        index1 = batch_index % T
        index1_end = index1 + batch_per_gpu
        # flatten the first two dimensions
        flat_tensor = out_list[index0].flatten(0, 1)

        return flat_tensor[index1:index1_end]

    def _step_target(self, epoch, batch_target, 
                     train=False, update_teacher=False, \
                     visualise=False, save_batch=False, writer=None, tag="train_target"):
        
        frames1, frames_gt, frames2, \
                affine, affine_inv = [self._prep_batch(t) for t in batch_target]

        # source forward pass
        losses, logits = self.net(frames1, frames_gt, frames2, \
                                  affine, affine_inv, use_teacher=True, \
                                  update_teacher=update_teacher, \
                                  T=self.cfg.TRAIN.GROUP_SIZE)

        if train:
            # Make sure we do not erase the gradient
            # from the pass of the source data
            if self.cfg.TRAIN.TARGET_ONLY:
                self.optim.zero_grad()

            # update on the target domain
            loss_target = self.cfg.MODEL.LR_TARGET * losses["self_ce"].mean()
            loss_target.backward()
            self.optim.step()

        if visualise:
            self._visualise(epoch, frames1, frames_gt, logits, writer, tag, image2=frames2)

        if save_batch:
            # Saving batch for visualisation
            self.save_fixed_batch(tag, batch_target)

        # reduce all for logging
        for key, val in losses.items():
            dist.all_reduce(val)
            val /= self.world_size
            losses[key] = losses[key].item()

        # only for convenience to compute validation IoU
        logits["mask_gt"] = frames_gt
        return losses, logits

    def train_epoch(self, epoch):

        stat = StatManager()
        stat.add_val("loss_ce")

        # adding stats for classes
        timer = Timer("[{}] New Epoch: ".format(self.gpu))
        step = partial(self.step, visualise=False)

        self.sampler_source.set_epoch(epoch)
        self.sampler_target.set_epoch(epoch)

        self.net.train()

        for i, (batch_source, batch_target) in enumerate(zip(self.loader_source, \
                                                             self.loader_target)):

            save_batch = i == 0

            #
            # Source pass
            #
            if not self.cfg.TRAIN.TARGET_ONLY:
                losses_src, _ = step(epoch, batch_source, train=True, \
                                     save_batch=save_batch, tag="train")

            #
            # Target pass
            #
            if self.cfg.MODEL.BASELINE:
                #
                # with ABN
                #
                # update only BN with target data
                with torch.no_grad():
                    losses, _ = self.step_target(epoch, batch_target, \
                                                 save_batch=save_batch, \
                                                 tag="train_target")
            else:
                #
                # with self-supervision
                #
                update_teacher = i % self.cfg.MODEL.NET_MOMENTUM_ITER == 0
                losses, _ = self.step_target(epoch, batch_target, \
                                                 update_teacher=update_teacher, \
                                                 save_batch=save_batch, \
                                                 tag="train_target")

            if not self.cfg.TRAIN.TARGET_ONLY:
                for key, val in losses_src.items():
                    losses["src_{}".format(key)] = val
                
            if self.main_process:

                for loss_key, loss_val in losses.items():
                    stat.update_stats(loss_key, loss_val)

                # intermediate logging
                if i % 10 == 0:
                    msg =  "Loss [{:04d}]: ".format(i)
                    for loss_key, loss_val in losses.items():
                        msg += " {} {:.4f} | ".format(loss_key, loss_val)
                    msg += " | Im/Sec: {:.1f}".format(i * self.cfg.TRAIN.BATCH_SIZE / timer.get_stage_elapsed())
                    print(msg)
                    sys.stdout.flush()
        
            if DEBUG and i > 20:
                break

        if self.main_process:

            for name, val in stat.items():
                print("{}: {:4.3f}".format(name, val))
                self.writer.add_scalar('train_epoch/{}'.format(name), val, epoch)

            # plotting learning rate
            for ii, l in enumerate(self.optim.param_groups):
                print("Learning rate [{}]: {:4.3e}".format(ii, l['lr']))
                self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)

        #if epoch % 10 == 0:
        if not self.cfg.TRAIN.TARGET_ONLY:
            self.visualise_results(epoch, self.writer, "train", self.step)

        #if not source_only:
        self.visualise_results(epoch, self.writer_target, "train_target", self.step_target)

    def validation(self, epoch, writer, loader, tag=None, step_func=None, max_iter=None):

        ignore_eval_ids = set(self.cfg.VAL.IGNORE_CLASS)
        print("Ignoring IDs from mIoU: ", ignore_eval_ids)

        if step_func is None:
            step_func = self.step

        if max_iter is None:
            max_iter = len(loader)

        stat = StatManager()

        if max_iter is None:
            max_iter = len(loader)

        # Fast test during the training
        def eval_batch(batch):

            loss, masks = step_func(epoch, batch, train=False, visualise=False)

            for loss_key, loss_val in loss.items():
                stat.update_stats(loss_key, loss_val)

            return masks

        self.net.eval()

        def pp_mask(logits_raw):
            return torch.argmax(logits_raw, 1)

        def ignore_indices(x):
            return torch.Tensor([v for i, v in enumerate(x) if not i in ignore_eval_ids])

        jaccard_stats = {}

        for n, batch in enumerate(loader):

            with torch.no_grad():
                masks_all = eval_batch(batch)

            # second element is assumed to be always GT masks
            masks_gt = batch[1]

            # for target data has shape [B,N,H,W]
            masks_gt = masks_gt.view(-1, *masks_gt.size()[-2:])

            for masks_layer, masks_raw in masks_all.items():

                if not masks_layer in ("logits_up", "teacher_init", "teacher_refined", "teacher_labels"):
                    continue

                if not masks_layer in jaccard_stats:
                    jaccard_stats[masks_layer] = Jaccard(self.nclass, self.gpu)

                if masks_layer in ("logits_up", "teacher_init", "teacher_refined"):
                    masks_pred = pp_mask(masks_raw)
                else:
                    masks_pred = masks_raw
                
                jaccard_stats[masks_layer].add_sample(masks_pred, masks_all["mask_gt"])


            if not tag is None and not self.has_fixed_batch(tag):
                self.save_fixed_batch(tag, batch)

            if n > max_iter:
                break

        checkpoint_score = 0.0

        # total classification loss
        if self.main_process:
            for stat_key, stat_val in stat.items():
                writer.add_scalar('all/{}'.format(stat_key), stat_val, epoch)
                print('Loss: {:4.3f}'.format(stat_val))

        #
        # segmentation
        #

        # gathering
        for mask_layer, jaccard_stat in jaccard_stats.items():
            for ni, className in enumerate(self.classNames):
                dist.all_reduce(jaccard_stat.tps[ni])
                dist.all_reduce(jaccard_stat.fns[ni])
                dist.all_reduce(jaccard_stat.fps[ni])

        # publishing
        if self.main_process:

            for mask_layer, jaccard_stat in jaccard_stats.items():

                print("Layer >>> ", mask_layer)
                jaccards, precision, recall = jaccard_stat.summarise()
                assert len(jaccards) == self.nclass

                for ni, className in enumerate(self.classNames):
                    label = "{}_{:02d}_{}".format(mask_layer, ni, className)
                    writer.add_scalar('%s/IoU' % label, jaccards[ni], epoch)
                    print("IoU_{}: {:4.3f}".format(className, jaccards[ni]))

                    writer.add_scalar('%s/Precision' % label, precision[ni], epoch)
                    print("Pr_{}: {:4.3f}".format(className, precision[ni]))

                    writer.add_scalar('%s/Recall' %  label, recall[ni], epoch)
                    print("Re_{}: {:4.3f}".format(className, recall[ni]))

                jaccards = ignore_indices(jaccards)
                precision = ignore_indices(precision)
                recall = ignore_indices(recall)

                print("Averaging {} classes: ".format(len(jaccards)))
                meanIoU = jaccards.mean().item()
                writer.add_scalar('%s_all/mIoU' % mask_layer, meanIoU, epoch)
                print('IoU: {:4.3f}'.format(meanIoU))

                meanPr = precision.mean().item()
                writer.add_scalar('%s_all/Precision' % mask_layer, meanPr, epoch)
                print(' Pr: {:4.3f}'.format(meanPr))

                meanRe = recall.mean().item()
                writer.add_scalar('%s_all/Recall' % mask_layer, meanRe, epoch)
                print(' Re: {:4.3f}'.format(meanRe))

                checkpoint_score = max(meanIoU, checkpoint_score)

        if not tag is None:
            self.visualise_results(epoch, writer, tag, step_func)

        return checkpoint_score

def main_worker(gpu, ngpus_per_node, argss, cfg):
    print("GPU / N ", gpu, ngpus_per_node)

    global args
    args = argss

    setproctitle.setproctitle("DA-SAC | {} | {}".format(args.run, gpu))

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    trainer = Trainer(args, cfg, gpu, ngpus_per_node)
    main_process = args.rank % ngpus_per_node == 0

    timer = Timer()
    def time_call(func, msg, *args, **kwargs):
        timer.reset_stage()
        val = func(*args, **kwargs)
        print("[{}] ".format(gpu) + msg + (" {:3.2}m".format(timer.get_stage_elapsed() / 60.)))
        return val

    log_val = 1 if DEBUG else cfg.LOG.ITER_VAL
    log_train = 1 if DEBUG else cfg.LOG.ITER_TRAIN
    log_target = 1 if DEBUG else cfg.LOG.ITER_TARGET

    for epoch in range(trainer.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):

        print("Epoch >>> {:02d} <<< ".format(epoch))
        if EVALUATE:
            if epoch > trainer.start_epoch and epoch % log_target == 0:
                time_call(trainer.validation, "Target / Train: ", epoch, trainer.writer_target, \
                          trainer.loader_target, tag="train_target", step_func=trainer.step_target, max_iter=300)

            if epoch > trainer.start_epoch and epoch % log_val == 0:
                scores = []

                for val_set, loader in trainer.valloaders.items():
                    score = time_call(trainer.validation, "Validation / {} /  Val: ".format(val_set), \
                                        epoch, trainer.writer_val[val_set], loader, tag=val_set)
                    if val_set == trainer.testset:
                        scores.append(score)

                if main_process and SNAPSHOT:
                    trainer.checkpoint_best(sum(scores), epoch)

            if EVALUATE_SOURCE and not cfg.TRAIN.TARGET_ONLY:
                if epoch > trainer.start_epoch and epoch % log_train == 0:
                    time_call(trainer.validation, "Validation / Train: ", epoch, trainer.writer, \
                              trainer.loader_source, tag="train", max_iter=100)

        # training 1 epoch
        time_call(trainer.train_epoch, "Train epoch: ", epoch)

def main():
    args = get_arguments(sys.argv[1:])

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.world_size = ngpus_per_node * args.world_size
    args.dist_url = "tcp://127.0.0.1:{}".format(find_free_port())

    print("World size: ", args.world_size, " / URL: {}".format(args.dist_url))
    print("# GPUs per node: ", ngpus_per_node)
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))

if __name__ == "__main__":
    main()
