"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import math
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.distributed as dist

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm
from utils.checkpoints import Checkpoint

class BaseTrainer(object):

    def __init__(self, args, cfg, main_process):
        self.args = args
        self.cfg = cfg
        self.start_epoch = 0
        self.best_score = -1e16
        self.checkpoint = Checkpoint(args.snapshot_dir, max_n = 3)
        self.main_process = main_process

        self.writer = None
        self.writer_target = None
        if main_process:
            logdir = os.path.join(args.logdir, 'train')
            self.writer = SummaryWriter(logdir)
            self.writer_target = SummaryWriter(os.path.join(args.logdir, 'train_target'))

    def checkpoint_best(self, score, epoch):

        if score > self.best_score:
            print(">>> Saving checkpoint with score {:3.2e}, epoch {}".format(score, epoch))
            self.best_score = score
            self.checkpoint.checkpoint(score, epoch)
            return True

        return False

    @staticmethod
    def get_optim(params, cfg):

        if not hasattr(torch.optim, cfg.OPT):
            print("Optimiser {} not supported".format(cfg.OPT))
            raise NotImplementedError

        optim = getattr(torch.optim, cfg.OPT)

        if cfg.OPT == 'Adam':
            print("Using Adam >>> learning rate = {:4.3e}, momentum = {:4.3e}, weight decay = {:4.3e}".format(cfg.LR, cfg.MOMENTUM, cfg.WEIGHT_DECAY))
            upd = torch.optim.Adam(params, lr=cfg.LR, \
                                   betas=(cfg.BETA1, 0.999), \
                                   weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPT == 'SGD':
            print("Using SGD >>> learning rate = {:4.3e}, momentum = {:4.3e}, weight decay = {:4.3e}".format(cfg.LR, cfg.MOMENTUM, cfg.WEIGHT_DECAY))
            upd = torch.optim.SGD(params, lr=cfg.LR, \
                                  momentum=cfg.MOMENTUM, \
                                  nesterov=cfg.OPT_NESTEROV, \
                                  weight_decay=cfg.WEIGHT_DECAY)

        else:
            upd = optim(params, lr=cfg.LR)

        upd.zero_grad()

        return upd
    
    def _visualise(self, epoch, image, masks_gt, logits, writer, tag, image2=None):

        # gathering
        def gather_cpu(tensor):
            out_list = [tensor.clone() for _ in range(self.world_size)]
            dist.all_gather(out_list, tensor)
            out_tensor = torch.cat(out_list, 0)
            return out_tensor.cpu()

        image = gather_cpu(image)
        masks_gt = gather_cpu(masks_gt)

        for key, val in logits.items():
            if not val.is_contiguous():
                print("Tensor {} is not contiguous".format(key))
                #val = val.contiguous()
            else:
                logits[key] = gather_cpu(val)

        if not image2 is None:
            image2 = gather_cpu(image2)

        data_palette = self.loader_source.dataset.get_palette()

        def downsize(x, mode="bilinear"):
            x = x.float()
            if x.dim() == 3:
                x = x.unsqueeze(1)
         
            if mode == "nearest":
                x = F.interpolate(x, self.cfg.TB.IM_SIZE, mode="nearest")
            else:
                x = F.interpolate(x, self.cfg.TB.IM_SIZE, mode=mode, align_corners=True)

            return x.squeeze()

        def compute_entpy_rgb(x):
            x = -(x*torch.log(1e-8 + x)).sum(1)
            x_min = x.min()
            x_max = x.max()
            x = (x - x_min) / (x_max - x_min)
            return self._error_rgb(x)

        visuals = []
        image_norm = downsize(self.denorm(image.clone())).cpu()
        visuals.append(image_norm)

        # GT mask
        masks_gt_rgb = downsize(self._apply_cmap(masks_gt, data_palette))
        masks_gt_rgb = 0.3 * image_norm + 0.7 * masks_gt_rgb
        visuals.append(masks_gt_rgb)

        if "teacher_labels" in logits:
            pseudo_gt = logits["teacher_labels"].cpu()
            masks_gt_rgb = downsize(self._apply_cmap(pseudo_gt, data_palette))
            masks_gt_rgb = 0.3 * image_norm + 0.7 * masks_gt_rgb
            visuals.append(masks_gt_rgb)

        # Prediction
        masks = downsize(F.softmax(logits["logits_up"], 1)).cpu()
        rgb_mask = self._mask_rgb(masks, image_norm, data_palette)

        masks_conf, masks_idx = masks.max(1)
        rgb_mask = self._apply_cmap(masks_idx, data_palette)
        rgb_mask = 0.3 * image_norm + 0.7 * rgb_mask
        visuals.append(rgb_mask)

        # Confidence
        masks_conf_rgb = self._error_rgb(1 - masks_conf, cmap=cm.get_cmap('inferno'))
        masks_conf_rgb = 0.3 * image_norm + 0.7 * masks_conf_rgb
        visuals.append(masks_conf_rgb)

        if image2 is not None:
            image2_norm = downsize(self.denorm(image2.clone())).cpu()
            visuals.append(image2_norm)

        vis_extra = []
        def vlogits_rgb(vlogits, frames_, softmax=True):
            if softmax:
                vlogits = F.softmax(vlogits, 1)

            masks = downsize(vlogits)
            masks_conf, masks_idx = masks.max(1)

            rgb_mask = self._apply_cmap(masks_idx, data_palette)
            rgb_mask = 0.3 * frames_ + 0.7 * rgb_mask
            vis_extra.append(rgb_mask)

            masks_conf_rgb = self._error_rgb(1 - masks_conf, cmap=cm.get_cmap('inferno'))
            masks_conf_rgb = 0.3 * frames_ + 0.7 * masks_conf_rgb
            vis_extra.append(masks_conf_rgb)

        if "teacher_init" in logits:
            # slow logits
            vlogits_rgb(logits["teacher_init"].cpu(), image2_norm)

        if "teacher_aligned" in logits:
            frames_aligned = downsize(self.denorm(logits["frames_aligned"].cpu()))
            vlogits_rgb(logits["teacher_aligned"].cpu(), frames_aligned, softmax=False)

        if "teacher_refined" in logits:
            logits_ = logits["teacher_refined"].cpu()
            vlogits_rgb(logits_.cpu(), image_norm, softmax=False)

        if "teacher_conf" in logits:
            teach_conf = downsize(logits["teacher_conf"].cpu())
            teach_conf_rgb = self._error_rgb((1. - teach_conf), cmap=cm.get_cmap('inferno'))
            teach_conf_rgb = 0.3 * image_norm + 0.7 * teach_conf_rgb
            visuals.append(teach_conf_rgb)

        visuals += vis_extra
        visuals = [x.float() for x in visuals]
        visuals = torch.cat(visuals, -1)

        if self.main_process:

            self._visualise_grid(writer, visuals, epoch, tag)

            if "running_conf" in logits:
                _,C,_,_ = logits["logits_up"].size()
                confs = logits["running_conf"].view(-1, C).mean(0).tolist()
                for ii, conf in enumerate(confs):
                    conf_key = "{:02d}".format(ii)
                    writer.add_scalar('running_conf/{}'.format(conf_key), conf, epoch)

    def save_fixed_batch(self, key, batch):

        if self.fixed_batch is None:
            self.fixed_batch = {}

        if key in self.fixed_batch:
            print("Updating fixed batch: ", key)

        self.fixed_batch[key] = {}
        batch_items = []
        for el in batch:
            el = el.clone().cpu() if torch.is_tensor(el) else el
            batch_items.append(el)

        self.fixed_batch[key] = batch_items

    def has_fixed_batch(self, key):
        return (not self.fixed_batch is None and \
                    key in self.fixed_batch)

    def _mask_rgb(self, masks, image_norm, palette, alpha=0.3):
        # visualising masks
        masks_conf, masks_idx = torch.max(masks, 1)
        masks_conf = masks_conf - F.relu(masks_conf - 1, 0)

        masks_idx_rgb = self._apply_cmap(masks_idx.cpu(), palette, mask_conf=masks_conf.cpu())
        return alpha * image_norm + (1 - alpha) * masks_idx_rgb

    def _apply_cmap(self, mask_idx, palette, mask_conf=None):

        # convert mask to RGB
        masks_rgb = []
        for mask in mask_idx.split(1, 0):
            mask = mask.cpu().numpy()[0].astype(np.uint32)
            im = Image.fromarray(mask).convert("P")
            im.putpalette(palette)
            mask_rgb = torch.as_tensor(np.array(im.convert("RGB")))
            mask_rgb = mask_rgb.permute(2,0,1)
            masks_rgb.append(mask_rgb[None, :, :, :])

        # cat back
        mask_rgb = torch.cat(masks_rgb, 0).float() / 255.0

        if not mask_conf is None:
            # entropy
            mask_entropy = 1 - mask_conf * torch.log(1e-8 + mask_conf) / (0.5 * math.log(1e-8 + 0.5))
            mask_rgb *= mask_entropy[:, None, :, :]

        return mask_rgb

    def _error_rgb(self, error_mask, cmap = cm.get_cmap('jet')):
        error_np = error_mask.cpu().numpy()

        # remove alpha channel
        error_rgb = cmap(error_np)[:, :, :, :3]
        error_rgb = np.transpose(error_rgb, (0,3,1,2))
        return torch.from_numpy(error_rgb)

    def _visualise_grid(self, writer, x_all, t, tag, ious=None, scores=None):

        # adding the labels to images
        bs, ch, h, w = x_all.size()
        x_all_new = torch.zeros(bs, ch, h, w)
        for b in range(bs):
            ndarr = x_all[b].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(ndarr)
            im_np = np.array(im).astype(np.float)
            x_all_new[b] = (torch.from_numpy(im_np)/255.0).permute(2,0,1)

        summary_grid = vutils.make_grid(x_all_new, nrow=1, padding=8, pad_value=0.9)
        writer.add_image(tag, summary_grid, t)

    def visualise_results(self, epoch, writer, tag, step_func):
        # visualising
        self.net.eval()
        with torch.no_grad():
            step_func(epoch, self.fixed_batch[tag], \
                      train=False, visualise=True, \
                      writer=writer, tag=tag)
