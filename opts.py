"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

from __future__ import print_function

import os
import sys
import torch
import argparse
from core.config import cfg

def add_global_arguments(parser):

    #
    # Model details
    #
    parser.add_argument("--snapshot-dir", type=str, default='./snapshots',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="Where to save log files of the model.")
    parser.add_argument("--exp", type=str, default="main",
                        help="ID of the experiment (multiple runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to snapshot to continue training from")
    parser.add_argument("--run", type=str, help="ID of the run")

    parser.add_argument('--mask-output-dir', type=str, default=None, help='path where to save masks')
    parser.add_argument('--split', type=str, default=None, help='[train|val|test] split to use')

    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')

    # 
    # Dataset details
    #
    parser.add_argument("--dataloader", type=str,
                        help="Specifies dataloader to use.")
    parser.add_argument("--infer-list", default="data/val_cityscapes.txt", type=str)

    #
    # Distributed Training
    #
    parser.add_argument('--world-size', default=-1, type=int,
                                help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://1.2.3.4:56789', type=str,
                                help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                                help='distributed backend')
    parser.add_argument('--rank', default=0, type=int,
                                help='node rank for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # Seed
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_global_arguments(args):

    args.cuda = torch.cuda.is_available()
    if not args.cuda:
        print("CPU mode is not supported.")
        sys.exit(1)

    args.logdir = os.path.join(args.logdir, args.dataloader, args.exp, args.run)
    maybe_create_dir(args.logdir)

    #
    # Model directories
    #
    args.snapshot_dir = os.path.join(args.snapshot_dir, args.dataloader, args.exp, args.run)
    maybe_create_dir(args.snapshot_dir)

def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Evaluation")

    add_global_arguments(parser)
    args = parser.parse_args(args_in)
    check_global_arguments(args)

    return args
