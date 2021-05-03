import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from models.basenet import BaseNet

BatchNorm = nn.SyncBatchNorm

class VGG16_FCN8s(BaseNet):

    def __init__(self, num_classes, criterion=None, pretrained=None, use_bn=False, freeze_bn=False, drop_rate=0.1):
        super().__init__()

        self.criterion = criterion

        #
        # Backbone
        #

        if use_bn:
            # VGG-16 w/ BN
            vgg = models.vgg16_bn()
            vgg = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vgg)

            # {23: 'pool3', 33: 'pool4'}
            self.block1 = vgg.features[:24]
            self.block2 = vgg.features[24:34]
            self.block3 = vgg.features[34:]
        else:
            # VGG-16 w/o BN
            vgg = models.vgg16()
            # {16: 'pool3', 23: 'pool4'}
            self.block1 = vgg.features[:17]
            self.block2 = vgg.features[17:24]
            self.block3 = vgg.features[24:]

        if not pretrained is None:
            print("VGG16-FCN8s: Loading snapshot: ", pretrained)
            vgg.load_state_dict(torch.load(pretrained))
        else:
            print("VGG16-FCN8s: Initialising from scratch")

        #
        # Head
        #
        if use_bn:
            self.vgg_head = nn.Sequential(
                nn.Conv2d(512, 4096, 7, padding=3), # we pad only here instead of input image
                BatchNorm(4096),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=drop_rate),
                nn.Conv2d(4096, 4096, 1),
                BatchNorm(4096),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=drop_rate),
                nn.Conv2d(4096, num_classes, 1)
            )
        else:
            self.vgg_head = nn.Sequential(
                nn.Conv2d(512, 4096, 7, padding=3), # we pad only here instead of input image
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=drop_rate),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=drop_rate),
                nn.Conv2d(4096, num_classes, 1)
            )

        # freezing the backbone
        if freeze_bn:
            # freezing the backbone
            self._freeze_bn(self)

        # adding from scratch layers
        self._from_scratch(self.vgg_head)

        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool4.weight.data.normal_(0, 0.01)

        self._from_scratch(self.score_pool4)

        #for param in self.score_pool4.parameters():
        #    # init.constant(param, 0)
        #    init.constant_(param, 0)


        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool3.weight.data.normal_(0, 0.01)

        self._from_scratch(self.score_pool3)

        #for param in self.score_pool3.parameters():
        #    # init.constant(param, 0)
        #    init.constant_(param, 0)

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 10.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 20.

    @staticmethod
    def up_x2(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

    def _backbone(self, x):

        # features
        pool3_s8 = self.block1(x)            # 1/8
        pool4_s16 = self.block2(pool3_s8)    # 1/16
        pool5_s32 = self.block3(pool4_s16)   # 1/32

        # bottleneck score at 1/32
        score = self.vgg_head(pool5_s32)     # 1/32

        # aggregating scores

        # adding pool4
        score4 = self.score_pool4(pool4_s16) # 1/16
        #score = self.up_x2(score) + 0.01 * score4   # 1/16
        score = self.up_x2(score) + score4   # 1/16

        # adding pool3
        score3 = self.score_pool3(pool3_s8)  # 1/8
        #score = self.up_x2(score) + 0.0001 * score3   # 1/8
        score = self.up_x2(score) + score3   # 1/8

        # output stride 8
        return score

    def forward(self, x, y=None):
        orig_size = x.size()[-2:]

        logits = self._backbone(x)
        logits_up = F.interpolate(logits, orig_size, mode="bilinear", align_corners=True)

        if y is None: # inference only
            return logits, logits_up

        losses = {}
        ce_loss = self.criterion(logits_up, y)
        losses["loss_ce"] = ce_loss.mean().view(1)

        return losses, {"logits_up": logits_up}

if __name__ == "__main__":
    net = VGG16_FCN8s(19, use_bn=True)
    inp = torch.randn(2, 3, 512, 512)
    #y = torch.argmax(torch.randn(2, 20, 512, 512), 1).cuda()
    x = net(inp)
    print(x.size())
