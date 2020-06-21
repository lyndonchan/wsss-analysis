import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.special import expit
import cv2
from misc import torchutils
from net import vgg16

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import m7
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CommonCAM(nn.Module):

    def __init__(self, tag, use_cls):
        super(CommonCAM, self).__init__()
        self.tag = tag
        self.use_cls = use_cls

    def _filter_adp_classes(self, x):
        adp_inds = [2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 21, 22, 23, 25, 26, 28, 29, 30, 32, 33, 35, 37,
                    38, 40, 45, 48, 49, 50]
        return x[adp_inds]

    def _adp_modify_morph(self, cam, img_orig):
        # Extract background activation
        background_max = 0.75
        adipose_inds = [18, 19, 20]

        mean_img = torch.mean(img_orig[0].float(), dim=2)
        sigmoid_input = 4 * (mean_img - 240)
        if torch.cuda.is_available():
            background_gradcam = background_max * expit(sigmoid_input.cpu().numpy())
        else:
            background_gradcam = background_max * expit(sigmoid_input.numpy())
        background_gradcam_arr = gaussian_filter(background_gradcam, sigma=2)
        background_gradcam_arr = cv2.resize(background_gradcam_arr, tuple(cam.shape[1:]))

        # Extract exempt activations
        adipose_cam, _ = torch.max(cam[adipose_inds], dim=0)
        if torch.cuda.is_available():
            background_gradcam = torch.relu(torch.from_numpy(background_gradcam_arr).cuda() - adipose_cam)
        else:
            background_gradcam = torch.relu(torch.from_numpy(background_gradcam_arr) - adipose_cam)

        # Insert background activation
        modified_cam = torch.cat((torch.unsqueeze(background_gradcam, 0), cam[self.use_cls, :]), dim=0)

        return modified_cam

    def _adp_modify_func(self, cam, img_orig):
        # Extract background activation
        background_max = 0.75
        background_exception_inds = [28, 29, 30]

        mean_img = torch.mean(img_orig[0].float(), dim=2)
        sigmoid_input = 4 * (mean_img - 240)
        if torch.cuda.is_available():
            background_gradcam = background_max * expit(sigmoid_input.cpu().numpy())
        else:
            background_gradcam = background_max * expit(sigmoid_input.numpy())
        background_gradcam_arr = gaussian_filter(background_gradcam, sigma=2)
        background_gradcam_arr = cv2.resize(background_gradcam_arr, tuple(cam.shape[1:]))

        # Subtract exempt activations
        func_cam, _ = torch.max(cam[background_exception_inds], dim=0)
        if torch.cuda.is_available():
            background_gradcam = torch.from_numpy(background_gradcam_arr).cuda() - func_cam
        else:
            background_gradcam = torch.from_numpy(background_gradcam_arr) - func_cam

        # Insert background activation
        modified_cam = torch.cat((torch.unsqueeze(background_gradcam, 0), cam[self.use_cls, :]), dim=0)

        # Extract other (non-functional) activation
        other_tissue_mult = 0.05
        adipose_inds = [18, 19, 20]
        other_moh = torch.max(modified_cam, dim=0)[0]
        other_gradcam = torch.unsqueeze(other_tissue_mult * (1 - other_moh), 0)
        adipose_gradcam = torch.unsqueeze(torch.max(cam[adipose_inds], dim=0)[0], 0)
        other_gradcam = torch.unsqueeze(torch.max(torch.cat((other_gradcam, adipose_gradcam), dim=0), dim=0)[0], 0)

        # Insert other (non-functional) activation
        modified_cam = torch.cat((torch.unsqueeze(modified_cam[0], 0), other_gradcam, modified_cam[1:]), dim=0)

        return modified_cam
