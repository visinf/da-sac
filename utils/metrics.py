"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import torch

class Jaccard:

    def __init__(self, num_classes, gpu, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tps = torch.zeros(num_classes).cuda(gpu)
        self.fps = torch.zeros(num_classes).cuda(gpu)
        self.fns = torch.zeros(num_classes).cuda(gpu)

    def add_sample(self, mask_pred, mask_gt):

        bs, h, w = mask_pred.size()
        assert bs == mask_gt.size(0), "Batch size mismatch"
        assert h == mask_gt.size(1), "Width mismatch"
        assert w == mask_gt.size(2), "Height mismatch"

        mask_pred = mask_pred.view(bs, 1, -1)
        mask_gt = mask_gt.view(bs, 1, -1)

        # ignore ambiguous
        ignore_mask = mask_gt == self.ignore_index
        mask_pred[ignore_mask] = self.ignore_index

        for label in range(self.num_classes):
            mask_pred_ = (mask_pred == label).float()
            mask_gt_ = (mask_gt == label).type_as(mask_pred_)

            self.tps[label] += (mask_pred_ * mask_gt_).sum().item()
            diff = mask_pred_ - mask_gt_
            self.fps[label] += diff.clamp(0).sum().item()
            self.fns[label] += (-diff).clamp(0).sum().item()

    def summarise(self):
        jaccards = torch.zeros(self.num_classes)
        precision = torch.zeros(self.num_classes)
        recall = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = self.tps[i]
            fn = self.fns[i]
            fp = self.fps[i]
            jaccards[i]  = tp / max(1e-3, fn + fp + tp)
            precision[i] = tp / max(1e-3, tp + fp)
            recall[i]    = tp / max(1e-3, tp + fn)

        return jaccards, precision, recall
