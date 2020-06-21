import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.special import expit
import cv2
from misc import torchutils
from net import vgg16
from net import common_cam

class CAM(common_cam.CommonCAM):

    def __init__(self, model_dir, dataset, tag, num_classes, use_cls):
        super(CAM, self).__init__(tag, use_cls)
        self.dataset = dataset
        if dataset in ['adp_morph', 'adp_func']:
            self.vgg16 = vgg16.vgg16_bn(model_dir, tag, num_classes, use_cls, batchnorm=False)
        else:
            self.vgg16 = vgg16.vgg16_bn(model_dir, tag, num_classes, use_cls, batchnorm=True)
        self.backbone = nn.ModuleList([self.vgg16.layer1, self.vgg16.layer2, self.vgg16.layer3, self.vgg16.layer4,
                                       self.vgg16.layer5])
        self.newly_added = nn.ModuleList([self.vgg16.classifier])

    def forward(self, x, x_orig=None):
        assert (self.dataset in ['adp_morph', 'adp_func']) ^ (x_orig is None)

        x = self.vgg16.layer1(x)
        x = self.vgg16.layer2(x)
        x = self.vgg16.layer3(x)
        x = self.vgg16.layer4(x)
        x = self.vgg16.layer5(x)

        # Classifier branch
        y = self.vgg16.avgpool(x)
        y = torch.flatten(y, 1)
        y = self.vgg16.classifier(y)[0]
        if torch.cuda.is_available():
            y_tmp = torch.ge(y, self.vgg16.thresholds.cuda())
        else:
            y_tmp = torch.ge(y, self.vgg16.thresholds)
        if self.dataset not in ['adp_morph', 'adp_func'] and torch.sum(y_tmp) == 0:
            y_tmp[torch.argmax(y)] = 1
        y = y_tmp
        if 'X1.7' in self.tag:
            y = self._filter_adp_classes(y)

        # CAM branch
        x = F.conv2d(x, self.vgg16.classifier[0].weight.unsqueeze(-1).unsqueeze(-1))
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        if 'X1.7' in self.tag:
            x = self._filter_adp_classes(x)

        # ADP modifications to CAM
        if self.dataset == 'adp_morph':
            x = self._adp_modify_morph(x, x_orig)
        elif self.dataset == 'adp_func':
            x = self._adp_modify_func(x, x_orig)

        return x, y