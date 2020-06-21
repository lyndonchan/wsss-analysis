
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from misc import imutils

import cv2

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def load_image_label_from_xml(img_name, dev_root):
    from xml.dom import minidom

    elem_list = minidom.parse(os.path.join(dev_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, dev_root):

    return [load_image_label_from_xml(img_name, dev_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def get_img_path(img_name, dev_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(dev_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    # img_name_list = np.loadtxt(dataset_path, dtype=np.int32)
    X = np.genfromtxt(dataset_path, delimiter='_', dtype=np.int32, usecols=[0, 1], unpack=False)
    img_name_list = np.array([np.int32(x[0]*1e6 + x[1]) for x in X])

    return img_name_list

class TorchvisionResize():
    def __init__(self, outsize=None):
        self.outsize = outsize

    def __call__(self, img):
        img = np.asarray(img, 'float64')
        if self.outsize is not None:
            outsize = tuple(self.outsize)
            if tuple(img.shape[:2]) != outsize:
                return cv2.resize(img, outsize)
        return img

class TorchvisionNormalize():
    def __init__(self, norm_mode='int'):
        self.norm_mode = norm_mode
        if self.norm_mode == 'int':
            self.mean = (104., 117., 123.)
            self.std = (255., 255., 255.)
        elif self.norm_mode == 'float':
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        elif self.norm_mode is not None:
            raise ValueError('norm_mode value is not \'int\' or \'float\'')
    def __call__(self, img):
        proc_img = np.empty_like(img, np.float32)
        img = np.float32(img)
        if self.norm_mode == 'int':
            proc_img[..., 0] = (img[..., 0] - self.mean[0]) / self.std[0]
            proc_img[..., 1] = (img[..., 1] - self.mean[1]) / self.std[1]
            proc_img[..., 2] = (img[..., 2] - self.mean[2]) / self.std[2]
        elif self.norm_mode == 'float':
            proc_img[..., 0] = (img[..., 0] / 255. - self.mean[0]) / self.std[0]
            proc_img[..., 1] = (img[..., 1] / 255. - self.mean[1]) / self.std[1]
            proc_img[..., 2] = (img[..., 2] / 255. - self.mean[2]) / self.std[2]
        elif self.norm_mode is None:
            return img
        else:
            raise ValueError('norm_mode value is not \'int\' or \'float\'')
        return proc_img

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, dev_root,
                 resize_long=None, rescale=None, img_resize=None,
                 norm_mode='int', hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.dev_root = dev_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_resize = img_resize
        self.img_normal = TorchvisionNormalize(norm_mode=norm_mode)
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.dev_root)))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_resize:
            img = self.img_resize(img)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            elif self.crop_method is not None:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img}

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, dev_root,
                 resize_long=None, rescale=None, outsize=None,
                 norm_mode='int', hor_flip=False,
                 crop_size=None, crop_method=None):
        img_resize = TorchvisionResize(outsize=outsize)
        super().__init__(img_name_list_path, dev_root,
                 resize_long, rescale, img_resize, norm_mode, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out

class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, dev_root, norm_mode='float',
                 outsize=None, scales=(1.0,)):
        self.scales = scales
        self.norm_mode = norm_mode
        self.outsize = outsize

        assert self.norm_mode in ['float', 'int']
        assert self.outsize in [(321, 321), (224, 224), None]

        super().__init__(img_name_list_path, dev_root, outsize=self.outsize,
                         norm_mode=self.norm_mode)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.dev_root))

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_resize(s_img)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out

class VOC12SegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir, crop_size, dev_root, outsize=None,
                 rescale=None, norm_mode='int', hor_flip=False,
                 crop_method='random'):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.dev_root = dev_root

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.outsize = outsize
        self.img_resize = TorchvisionResize(outsize=self.outsize)
        self.img_normal = TorchvisionNormalize(norm_mode=norm_mode)
        self.hor_flip = hor_flip
        self.crop_method = crop_method

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.dev_root))
        label = imageio.imread(os.path.join(self.label_dir, name_str + '.png'))

        img = np.asarray(img)

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_resize:
            img = self.img_resize(img)
            label = self.img_resize(label)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, 255))
        elif self.crop_method is not None:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        img = imutils.HWC_to_CHW(img)

        return {'name': name, 'img': img, 'label': label}

class VOC12AffinityDataset(VOC12SegmentationDataset):
    def __init__(self, img_name_list_path, label_dir, crop_size, dev_root,
                 indices_from, indices_to, rescale=None,
                 norm_mode='int', hor_flip=False, crop_method=None, outsize=None):
        super().__init__(img_name_list_path, label_dir, crop_size, dev_root, outsize, rescale, norm_mode,
                         hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out

