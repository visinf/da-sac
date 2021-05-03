"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0

Description: Single-scale inference
"""

import os
import sys
import numpy as np
import imageio
import time

import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from models import get_model
from utils.checkpoints import Checkpoint
from utils.timer import Timer
from datasets import get_num_classes
from datasets.dataloader_infer import get_dataloader 
from utils.sys_tools import check_dir
from utils.palette import *
from tools.category import labels as CS_LABELS

# deterministic inference
from torch.backends import cudnn

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

# save logits
D_SAVE_RAW = False
# save CS labels
D_SAVE_CS = True
D_VIS = False
# test mode: no ground truth
D_NOGT = False

def mask2rgb(mask, palette):
    im = Image.fromarray(mask).convert("P")
    im.putpalette(palette)
    mask_rgb = np.array(im.convert("RGB"))
    return mask_rgb / 255.


def mask_overlay(mask, image, palette):
    """Creates an overlayed mask visualisation"""
    mask_rgb = mask2rgb(mask, palette)
    return 0.3 * image + 0.7 * mask_rgb

def convert_to_cs(labels):
    """Convert train IDs to Cityscapes IDs"""
    labels_cs = np.zeros_like(labels)
    for label in CS_LABELS:
        labels_cs[labels == label.trainId] = label.id
    return labels_cs

class ResultWriter:
    
    def __init__(self, palette, out_path, verbose=False, raw=False, save_cs=False):
        self.palette = palette
        self.out_path = out_path
        self.verbose = verbose
        self.raw = raw
        self.save_cs = save_cs

    def save(self, image, gt_mask, masks, im_name):

        masks_raw = masks.numpy()
        pred = np.argmax(masks_raw, 0).astype(np.uint8)
        filepath = os.path.join(self.out_path, im_name + '.png')
        imageio.imwrite(filepath, pred)

        if self.save_cs:
            pred_cs = convert_to_cs(pred)
            #im_name_cs = im_name.replace("labelTrainIds", "labelIds")
            filepath = os.path.join(self.out_path, "cs", im_name + '.png')
            imageio.imwrite(filepath, pred_cs)

        if self.raw:
            filepath = os.path.join(self.out_path, "raw", im_name)
            np.savez_compressed(filepath, raw=masks_raw)

        if self.verbose:

            if D_NOGT:
                masks = pred
                images = image.numpy()
            else:
                mask_gt = gt_mask.numpy().astype(np.uint8)
                masks = np.concatenate([pred, mask_gt], 1)
                images = np.concatenate([image, image], 2)

            images = np.transpose(images, [1,2,0])

            overlay = mask_overlay(masks, images, palette)
            filepath = os.path.join(self.out_path, "vis", im_name + '.png')
            imageio.imwrite(filepath, (overlay * 255.).astype(np.uint8))

def convert_dict(state_dict):
    new_dict = {}
    for k,v in state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict

if __name__ == '__main__':

    # loading the model
    args = get_arguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # initialising the dirs
    check_dir(args.mask_output_dir, "vis")
    check_dir(args.mask_output_dir, "cs")
    if D_SAVE_RAW:
        check_dir(args.mask_output_dir, "raw")

    #check_dir(args.mask_output_dir, "crf")

    num_classes = get_num_classes(args)

    # Loading the model
    model = get_model(cfg.MODEL, 0, num_classes=num_classes)
    assert os.path.isfile(args.resume), "Snapshot not found: {}".format(args.resume)
    state_dict = convert_dict(torch.load(args.resume)["model"])
    print(model)
    model.load_state_dict(state_dict, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    # setting the evaluation mode
    model.eval()
    model = nn.DataParallel(model).cuda()

    infer_dataset = get_dataloader(args.dataloader, cfg, args.infer_list)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.workers, \
                                   pin_memory=True, batch_size=1)
    palette = infer_dataset.get_palette()

    timer = Timer()
    N = len(infer_data_loader)

    pool = mp.Pool(processes=args.workers)
    writer = ResultWriter(palette, args.mask_output_dir, verbose=D_VIS, raw=D_SAVE_RAW, save_cs=D_SAVE_CS)

    for iter, (image, gt_mask, im_name) in enumerate(tqdm(infer_data_loader)):
        
        with torch.no_grad():
            _, logits = model(image, teacher=False)
            masks_pred = F.softmax(logits, 1)

        image = infer_dataset.denorm(image)
        #writer.save(image[0], gt_mask[0], masks_pred[0].cpu(), im_name[0])
        pool.apply_async(writer.save, args=(image[0], gt_mask[0], masks_pred[0].cpu(), im_name[0]))

        timer.update_progress(float(iter + 1) / N)

        if iter % 50 == 0:
            msg = "Finish time: {}".format(timer.str_est_finish())
            tqdm.write(msg)
            sys.stdout.flush()
            if D_VIS and iter > 0:
                print("\nSleeping 60 seconds...")
                time.sleep(90.)

    pool.close()
    pool.join()
