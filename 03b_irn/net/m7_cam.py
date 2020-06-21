import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import m7
import numpy as np
import cv2
import matplotlib.pyplot as plt
from net import common_cam

class CAM(common_cam.CommonCAM):

    def __init__(self, model_dir, dataset, tag, num_classes, use_cls):
        super(CAM, self).__init__(tag, use_cls)
        self.dataset = dataset
        if dataset in ['adp_morph', 'adp_func'] and 'X1.7' in tag:
            num_classes = 51
        self.m7 = m7.m7(model_dir, tag, num_classes, use_cls, batchnorm=True)
        self.backbone = nn.ModuleList([self.m7.layer1, self.m7.layer2, self.m7.layer3_p1, self.m7.layer3_p2])
        self.newly_added = nn.ModuleList([self.m7.classifier])

    def forward(self, x, x_orig=None):
        assert (self.dataset in ['adp_morph', 'adp_func']) ^ (x_orig is None)

        x = self.m7.layer1(x)
        x = self.m7.layer2(x)
        x = self.m7.layer3_p1(x)

        # Classifier branch
        y = self.m7.layer3_p2(x)
        y = self.m7.maxpool(y)
        y = torch.flatten(y, 1)
        y = self.m7.classifier(y)[0]
        if torch.cuda.is_available():
            y_tmp = torch.ge(y, self.m7.thresholds.cuda())
        else:
            y_tmp = torch.ge(y, self.m7.thresholds)
        if self.dataset not in ['adp_morph', 'adp_func'] and torch.sum(y_tmp) == 0:
            y_tmp[torch.argmax(y)] = 1
        y = y_tmp
        if 'X1.7' in self.tag:
            y = self._filter_adp_classes(y)

        # Grad-CAM branch
        weights = torch.from_numpy(self.m7.gradcam_weights).float().cuda()
        x = F.conv2d(x, weights.transpose(1,0).unsqueeze(-1).unsqueeze(-1))
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