"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import grad

from models.basenet import BaseNet

class SAC_Baseline(BaseNet):

    def __init__(self, cfg, backbone, rank, **kwargs):
        super(SAC_Baseline, self).__init__()

        self.backbone = backbone

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            print("World size: ", self.world_size)
        else:
            # inference
            self.world_size = 1

        self.rank = rank

        if "criterion" in kwargs:
            self.criterion = kwargs["criterion"]

    def forward(self, x=None, y=None, x2=None, use_teacher=False, update_teacher=False):
        return self.backbone(x, y)

    def parameter_groups(self, base_lr, wd):
        return self.backbone.parameter_groups(base_lr, wd)
    

class SAC(SAC_Baseline):

    def __init__(self, cfg, backbone, slow_copy, rank, **kwargs):
        super(SAC, self).__init__(cfg, backbone, rank, **kwargs)

        self.cfg = cfg
        self.backbone = backbone

        self.pool_func = self._get_op(cfg.CONF_POOL)
        self.loss_func = self._get_op(cfg.LOSS)

        # moving class prior
        running_conf = torch.zeros(kwargs["num_classes"])
        self.register_buffer("running_conf", running_conf)

        # initialising slow net
        self.slow_net = slow_copy
        self.slow_net.eval()
        for p in self.slow_net.parameters():
            p.requires_grad = False

        #self.slow_init = False
        self.register_buffer("slow_init", torch.Tensor([False]))

    def _get_op(self, name):
        op_name = "_{}".format(name)
        assert hasattr(self, op_name), "Pooling OP {} not found".format(op_name)
        return getattr(self, op_name)

    @torch.no_grad()
    def _momentum_update(self, update=False):
        """Momentum update"""

        # initialising
        if not self.slow_init[0]:
            print(">>> Re-initialising ")

            self.running_conf.fill_(self.cfg.THRESHOLD_BETA)
            self.slow_init[0] = True
            self.slow_net.load_state_dict(self.backbone.state_dict())
            return torch.Tensor([0.]).type_as(self.running_conf)

        # parameter distance
        diff_sum = 0.
        slow_net_dict = self.slow_net.state_dict()
        backbone_dict = self.backbone.state_dict()
        for key, val in backbone_dict.items():

            if key.split(".")[-1] in ("weight", "bias", "running_mean", "running_var"):

                dist = torch.norm(slow_net_dict[key] - val)

                if update:
                    slow_net_dict[key].mul_(self.cfg.NET_MOMENTUM)
                    slow_net_dict[key].add_(val * (1. - self.cfg.NET_MOMENTUM))

                    #if dist > 10.:
                    #    print("Large update for: ", key, dist)

                diff_sum += dist

        return diff_sum.view(1)

    @torch.no_grad()
    def _update_running_conf(self, probs, tolerance=1e-8):
        """Maintain the moving class prior"""
        B,C,H,W = probs.size()
        probs_avg = probs.mean(0).view(C,-1).mean(-1)

        # updating the new records: copy the value
        update_index = probs_avg > tolerance
        new_index = update_index & (self.running_conf == self.cfg.THRESHOLD_BETA)
        self.running_conf[new_index] = probs_avg[new_index]

        # use the moving average for the rest
        self.running_conf *= self.cfg.STAT_MOMENTUM
        self.running_conf += (1 - self.cfg.STAT_MOMENTUM) * probs_avg

    def _focal_ce(self, logits, pseudo_gt, teacher_probs, p = 3):
        focal_weight = (1 - self.running_conf.clamp(0.)) ** p
        loss_ce = F.cross_entropy(logits, pseudo_gt, weight=focal_weight, ignore_index=255, reduction="none")

        with torch.no_grad():
            C = logits.size(1)
            B,H,W = loss_ce.size()
            loss_per_class = torch.zeros_like(logits)
            loss_idx = pseudo_gt.clone()
            loss_idx[pseudo_gt == 255] = 0
            loss_per_class.scatter_(1, loss_idx[:,None], loss_ce[:,None]) # B,C,H,W
            loss_per_class = loss_per_class.view(B, C, -1).mean(-1).mean(0)

        return loss_ce.mean(), loss_per_class

    def _focal_ce_conf(self, logits, pseudo_gt, teacher_probs, p = 3):
        focal_weight = (1 - self.running_conf.clamp(0.)) ** p
        loss_ce = F.cross_entropy(logits, pseudo_gt, weight=focal_weight, ignore_index=255, reduction="none")

        with torch.no_grad():
            C = logits.size(1)
            B,H,W = loss_ce.size()
            loss_per_class = torch.zeros_like(logits)
            loss_idx = pseudo_gt.clone()
            loss_idx[pseudo_gt == 255] = 0
            loss_per_class.scatter_(1, loss_idx[:,None], loss_ce[:,None]) # B,C,H,W
            loss_per_class = loss_per_class.view(B, C, -1).mean(-1).mean(0)

        #teacher_norm = teacher_probs.sum() + 1e-3
        loss = (loss_ce * teacher_probs).mean() # / teacher_norm
        return loss, loss_per_class

    def _threshold_discount(self):
        return 1. - torch.exp(- self.running_conf / self.cfg.THRESHOLD_BETA)

    @torch.no_grad()
    def _pseudo_labels_probs(self, probs, ignore_augm, discount = True):
        """Consider top % pixel w.r.t. each image"""

        B,C,H,W = probs.size()
        max_conf, max_idx = probs.max(1, keepdim=True) # B,1,H,W

        probs_peaks = torch.zeros_like(probs)
        probs_peaks.scatter_(1, max_idx, max_conf) # B,C,H,W
        top_peaks, _ = probs_peaks.view(B,C,-1).max(-1) # B,C

        # > used for ablation
        #top_peaks.fill_(1.)

        top_peaks *= self.cfg.RUN_CONF_UPPER

        if discount:
            # discount threshold for long-tail classes
            top_peaks *= self._threshold_discount().view(1, C)

        top_peaks.clamp_(self.cfg.RUN_CONF_LOWER) # in-place
        probs_peaks.gt_(top_peaks.view(B,C,1,1))

        # ignore if lower than the discounted peaks
        ignore = probs_peaks.sum(1, keepdim=True) != 1

        # thresholding the most confident pixels
        pseudo_labels = max_idx.clone()
        pseudo_labels[ignore] = 255

        pseudo_labels = pseudo_labels.squeeze(1)
        pseudo_labels[ignore_augm] = 255

        return pseudo_labels, max_conf, max_idx

    def _entropy(self, probs, eps = 1e-5):
        probs_eps = (probs + eps) / (1 + eps)
        entropy = -(probs * torch.log(probs_eps)).sum(1, keepdim=True)

        probs_zero = probs.sum(1, keepdim=True)
        entropy[probs_zero < 0.1] = 1. / eps

        return entropy

    @torch.no_grad()
    def _gather(self, tensor, T):
        """Gather if the images were previously 
        scattered across GPUs"""

        B = tensor.size(0)
        stride = max(1, T // B) # how many tensors to gather
        if stride > 1:
            # we need to gather
            out_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(out_list, tensor)

            # select which tensors to concatenate
            index = stride * (self.rank * B // T) # 0,1,2,3 -> 0,0,2,2
            index_end = index + stride # 0,0,2,2 -> 2,2,4,4
            
            return torch.cat(out_list[index:index_end], 0)

        return tensor

    @torch.no_grad()
    def _minentropy_pool(self, probs, T, tolerance=0.1):
        # compute entropy
        BT,C,H,W = probs.size()
        entropy = self._entropy(probs)
        entropy_T = entropy.view(-1,T,1,H,W)

        # B,1,1,H,W
        entropy_min = entropy_T.argmin(1, keepdim=True)
        # B,1,C,H,W
        entropy_min = entropy_min.expand(-1,-1,C,-1,-1)

        # select predictions with min. entropy
        probs_T = probs.view(-1,T,C,H,W)
        masks = probs_T.sum(1, keepdim=True).sum(2, keepdim=True) > tolerance

        probs_T[:,:,:,:,:] = probs_T.gather(1, entropy_min)
        masks = masks.expand(-1,T,-1,-1,-1).type_as(probs_T)
        return probs_T.view(BT,C,H,W), masks.view(BT,1,H,W)

    @torch.no_grad()
    def _avg_pool(self, probs, T, tolerance=0.1):
        # T0 is a fractional inner batch size:
        # for example, if only 2 images fit on one GPU, but
        # we have 4 samples per target image, then T0 = 2,
        # i.e. each GPU processes only 2 out of 4 samples
        T0 = min(T, probs.size(0))

        probs = self._gather(probs, T)

        # compute entropy
        _,C,H,W = probs.size()

        # select predictions with min. entropy
        probs_T = probs.view(-1,T,C,H,W)
        probs_T_avg = probs_T.sum(1, keepdim=True) # B,1,C,H,W

        # count valid predictions
        probs_T_sum = probs_T_avg.sum(2, keepdim=True) # B,1,1,H,W

        # at least 2 predictions [B,1,1,H,W]
        #mask = (probs_T_sum - 1.).clamp(0, 1)
        mask = (probs_T_sum > tolerance).type_as(probs)

        # averaging [B,1,C,H,W]
        probs_T_avg /= probs_T_sum.clamp(1e-3)

        # normalising the shape
        probs_T_avg = probs_T_avg.expand(-1,T0,-1,-1,-1) # B,T0,C,H,W
        mask = mask.expand(-1,T0,-1,-1,-1)

        return probs_T_avg.flatten(0,1), mask.flatten(0,1)

    @torch.no_grad()
    def _refine(self, frames, pred_logits, T, affine, affine_inv, ignore_mask, pool=True, debug=True):
        _,C,h,w = frames.size()

        pred_logits = F.interpolate(pred_logits, (h, w), mode="bilinear", align_corners=True)
        pred_probs = F.softmax(pred_logits, 1)

        if self.training:
            self._update_running_conf(pred_probs)

        # ignoring predictions in the image paddings
        pred_probs *= 1 - ignore_mask[:, None].type_as(pred_probs)

        diags = {}
        if not pool:
            return pred_probs, diags

        # probs: warp to the reference
        affine_grid_probs = F.affine_grid(affine, size=pred_probs.size(), align_corners=False)
        pred_probs_aligned = F.grid_sample(pred_probs, affine_grid_probs, align_corners=False)

        # diagnostics
        diags["teacher_aligned"] = pred_probs_aligned
        if debug:
            affine_frames = F.affine_grid(affine, size=frames.size(), align_corners=False)
            diags["frames_aligned"] = F.grid_sample(frames, affine_frames, align_corners=False)

        ### compute valid mask
        valid_probs = torch.ones_like(pred_probs_aligned)
        affine_grid_inv = F.affine_grid(affine_inv, size=valid_probs.size(), align_corners=False)
        valid_aligned = F.grid_sample(valid_probs, affine_grid_inv, align_corners=False)

        ####
        # Pool predictions
        refined_aligned, valid = self.pool_func(pred_probs_aligned * valid_aligned, T)
        ###

        # warping back
        refined = F.grid_sample(refined_aligned, affine_grid_inv, align_corners=False)
        refined_valid = F.grid_sample(valid, affine_grid_inv, align_corners=False)
        refined *= refined_valid

        return refined, diags

    def forward(self, x, y=None, x2=None, affine=None, affine_inv=None, \
                      use_teacher=False, update_teacher=False, reset_teacher=False, T=None, teacher=False):
        """Args:
                x: input images [BxCxHxW]
                y: ground-truth for source images [BxHxW]
                x2: input images w/o photometric noise [BxCxHxW]
                T: length of the sequences
        """

        if y is None:
            # inference-only mode
            if teacher:
                return self.slow_net(x)
            else:
                return self.backbone(x)

        if reset_teacher:
            self.slow_init[0] = False

        # note that this extracts not the original ambiguous 
        # pixels in the ground-truth label, but only the pixels
        # that we added via data augmentation (e.g. padding)
        ignore_mask = (y == -1)
        y[ignore_mask] = 255

        losses, net_outs = self.backbone(x, y)

        if update_teacher:
            print("Updating the teacher")
            losses["teacher_diff"] = self._momentum_update(True)

        if use_teacher:

            self.slow_net.eval()
            with torch.no_grad():
                slow_logits, slow_logits_up = self.slow_net(x2)

            # multi-scale fusion
            probs_teacher, diags = self._refine(x2, slow_logits, T, affine, affine_inv, ignore_mask, pool=self.cfg.CONF_POOL_ON)
            probs_teacher = probs_teacher.detach()

            # generating pseudo labels
            pseudo_labels, teacher_conf, teacher_maxi = self._pseudo_labels_probs(probs_teacher, ignore_mask, self.cfg.CONF_DISCOUNT)

            # computing the loss
            loss_ce, loss_per_class = self.loss_func(net_outs["logits_up"], pseudo_labels, teacher_conf, self.cfg.FOCAL_P)
            losses["self_ce"] = loss_ce.mean().view(1)

            net_outs["teacher_init"] = slow_logits_up.detach()
            net_outs["teacher_refined"] = probs_teacher.detach()
            net_outs["teacher_conf"] = teacher_conf.detach()

            # CE per class
            # uncomment to print
            #for i,v in enumerate(loss_per_class):
            #    losses["self_ce_{}".format(i)] = v

            net_outs["teacher_labels"] = pseudo_labels.detach()
            net_outs["running_conf"] = self.running_conf
            losses["teacher_diff"] = self._momentum_update(False)

            net_outs.update(diags)

        return losses, net_outs

    def parameter_groups(self, base_lr, wd):
        return self.backbone.parameter_groups(base_lr, wd)
