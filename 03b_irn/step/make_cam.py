import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

from keras.models import model_from_json
import keras.backend as K

import numpy as np
import importlib
import os
import cv2
import imageio
from tqdm import tqdm

import voc12.dataloader
import adp.dataloader
import deepglobe.dataloader
from misc import torchutils, imutils
import matplotlib.pyplot as plt

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        with tqdm(total=len(data_loader)) as pbar:
            for iter, pack in enumerate(data_loader):

                img_name = pack['name'][0]
                size = pack['size']

                strided_size = imutils.get_strided_size(size, 4)
                strided_up_size = imutils.get_strided_up_size(size, 16)

                if args.dataset in ['adp_morph', 'adp_func']:
                    outputs, labels = zip(*[model(img.cuda(non_blocking=True), orig_img.cuda(non_blocking=True)) for
                                            img, orig_img in zip(pack['img'], pack['orig_img'])])
                else:
                    outputs, labels = zip(*[model(img.cuda(non_blocking=True)) for img in pack['img']])
                if 'train' in args.split:
                    label = pack['label'][0]
                else:
                    label = labels[0][args.use_cls]

                valid_cat = torch.nonzero(label)[:, 0]
                if args.dataset in ['adp_morph', 'adp_func']:
                    if torch.cuda.is_available():
                        valid_cat = torch.cat((torch.from_numpy(np.array(range(len(args.class_names['bg'])),
                                             dtype=np.int64)).cuda(), valid_cat.cuda() + len(args.class_names['bg'])))
                    else:
                        valid_cat = torch.cat((torch.from_numpy(np.array(range(len(args.class_names['bg'])),
                                             dtype=np.int64)), valid_cat + len(args.class_names['bg'])))

                if len(valid_cat) > 0:
                    strided_cam = torch.sum(torch.stack(
                        [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                         in outputs]), 0)

                    highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                                 mode='bilinear', align_corners=False) for o in outputs]
                    highres_cam = torch.sum(torch.stack(tuple(highres_cam), 0), 0)[:, 0, :size[0], :size[1]]

                    strided_cam = strided_cam[valid_cat]
                    strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

                    highres_cam = highres_cam[valid_cat]
                    highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

                    # save cams
                    if args.dataset not in ['deepglobe', 'deepglobe_balanced']:
                        np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                                {"keys": valid_cat.cpu().numpy(), "cam": strided_cam.cpu().numpy(),
                                 "high_res": highres_cam.cpu().numpy()})
                    else:
                        np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                                {"keys": valid_cat.cpu().numpy(), "cam": strided_cam.cpu().numpy()})
                else:
                    np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                            {"keys": np.empty(0), "cam": np.empty(0), "high_res": np.empty(0)})
                pbar.update(1)
                # plt.imshow(highres_cam.cpu().numpy()[0])
                # plt.show()
                # if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                #     print("%d " % ((5*iter+1)//(len(databin) // 20))) # , end='')

def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')(args.model_dir, args.dataset, args.tag,
                                                                      args.num_classes, args.use_cls)
    if args.model_id == 'resnet50':
        model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    if args.dataset == 'voc12':
        dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.val_list, norm_mode=args.norm_mode,
                                                                 outsize=args.outsize, dev_root=args.dev_root,
                                                                 scales=args.cam_scales)
    elif args.dataset in ['adp_morph', 'adp_func']:
        dataset = adp.dataloader.ADPClassificationDatasetMSF(args.val_list, norm_mode=args.norm_mode,
                                                             outsize=args.outsize, dev_root=args.dev_root,
                                                             htt_type=args.dataset.split('_')[-1],
                                                             is_eval=args.split == 'evaluation', scales=args.cam_scales)
    elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
        dataset = deepglobe.dataloader.DeepGlobeClassificationDatasetMSF(args.val_list, norm_mode=args.norm_mode,
                                                                     outsize=args.outsize, dev_root=args.dev_root,
                                                                     is_balanced=args.dataset == 'deepglobe_balanced',
                                                                     scales=args.cam_scales)
    else:
        raise KeyError('Dataset %s not yet implemented' % args.dataset)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)

    torch.cuda.empty_cache()