"""Code based on
https://raw.githubusercontent.com/yzou2/CRST/fce34003dd29c5f12f39d1c228dacd6277a064ae/deeplab/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from models.basenet import BaseNet
affine_par = True

__all__ = ['Res_Deeplab']

BatchNorm = nn.SyncBatchNorm

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = BatchNorm(planes,affine = affine_par)
        #for i in self.bn1.parameters():
        #    i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = BatchNorm(planes,affine = affine_par)
        #for i in self.bn2.parameters():
        #    i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4, affine = affine_par)
        #for i in self.bn3.parameters():
        #    i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, fan_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(fan_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.layer5 = Classifier_Module(2048, [6,12,18,24], [6,12,18,24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion,affine = affine_par))
        #for i in downsample._modules['1'].parameters():
        #    i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

class DeepLabV2_ResNet101(BaseNet):

    def __init__(self, num_classes=20, \
                    criterion=nn.CrossEntropyLoss(ignore_index=255, reduction="none"), \
                    pretrained=None, freeze_bn=False):
        super().__init__()

        self.model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

        # converting to SyncBatchNorm
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if not pretrained is None:
            self._init_weights(pretrained)
        else:
            print("ResNet-101: Starting training from scratch")

        # fixing BN of the backbone
        if freeze_bn:
            print("DeepLabv2/ResNet-101: Fixing BN")
            self._freeze_bn(self)

        self._from_scratch(self.model.layer5)
        self.criterion = criterion

    def _init_weights(self, path_to_weights):
        print("Loading weights from: ", path_to_weights)
        weights_dict = torch.load(path_to_weights)
        self.model.load_state_dict(weights_dict, strict=False)

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 10.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 20.

    def forward(self, im, y=None):
        orig_size = im.size()[-2:]

        logits = self.model(im)
        logits_up = F.interpolate(logits, orig_size, mode="bilinear", align_corners=True)

        if y is None: # inference only
            return logits, logits_up

        losses = {}
        ce_loss = self.criterion(logits_up, y)
        losses["loss_ce"] = ce_loss.mean().view(1)

        return losses, {"logits_up": logits_up,
                        "logits": logits}

class DeepLabV2_VGG16(BaseNet):

    def __init__(self, num_classes, criterion=None, pretrained=None, use_bn=False, freeze_bn=False):
        super(DeepLabV2_VGG16, self).__init__()

        self.criterion = criterion

        if use_bn:
            # VGG-16 with BN
            vgg = models.vgg16_bn()
            add_dilation = (34, 37, 40)
            remove_pool = (33, 43)
        else:
            # VGG-16 without BN
            vgg = models.vgg16()
            add_dilation = (24, 26, 28)
            remove_pool = (23, 30)

        vgg = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vgg)

        if not pretrained is None:
            print("VGG16: Loading snapshot: ", pretrained)
            vgg.load_state_dict(torch.load(pretrained))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        for i in add_dilation:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        # remove pool4/pool5
        features = [f for i,f in enumerate(features) if not i in remove_pool]

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)
        features += [fc6, nn.ReLU(inplace=True), \
                     fc7, nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*features)

        #self.bn = nn.SyncBatchNorm(1024)
        self.classifier = Classifier_Module(1024, [6,12,18,24], [6,12,18,24], num_classes)

        if freeze_bn:
            # freezing the backbone
            print("DeepLabv2/VGG-16: Fixing BN")
            self._freeze_bn(self)

        # marking layers as new
        self._from_scratch(self.classifier)
        self._from_scratch(fc6)
        self._from_scratch(fc7)

    def lr_mult(self):
        """Learning rate multiplier for weights.
        Returns: [old, new]"""
        return 1., 10.

    def lr_mult_bias(self):
        """Learning rate multiplier for bias.
        Returns: [old, new]"""
        return 2., 20.

    def _backbone(self, x):
        x = self.features(x)
        #x = F.relu(self.bn(x), inplace=True)
        x = self.classifier(x)
        return x

    def forward(self, im, y=None):
        orig_size = im.size()[-2:]

        logits = self._backbone(im)
        logits_up = F.interpolate(logits, orig_size, mode="bilinear", align_corners=True)

        if y is None: # inference only
            return logits, logits_up

        losses = {}
        ce_loss = self.criterion(logits_up, y)
        losses["loss_ce"] = ce_loss.mean().view(1)

        return losses, {"logits_up": logits_up, \
                        "logits": logits}

if __name__ == "__main__":

    model = DeepLabV2_ResNet101().cuda()

    x = torch.randn(8,3,1024,1024).cuda()
    y = torch.randn(8,20,1024,1024).argmax(1).cuda()

    print(x.size())
    with torch.no_grad():
        losses, outs = model(x, y)
    print("Done")
