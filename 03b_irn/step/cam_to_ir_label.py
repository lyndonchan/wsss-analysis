
import os
import numpy as np
import cv2
import imageio
from tqdm import tqdm

from torch import multiprocessing
from torch.utils.data import DataLoader

import voc12.dataloader
import adp.dataloader
import deepglobe.dataloader
from misc import torchutils, imutils
import matplotlib.pyplot as plt


def _work(process_id, infer_dataset, args):

    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    with tqdm(total=len(infer_data_loader)) as pbar:
        for iter, pack in enumerate(infer_data_loader):
            img_name = pack['name'][0] # voc12.dataloader.decode_int_filename(pack['name'][0])
            img = pack['img'][0].numpy()
            cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

            if args.dataset in ['adp_morph', 'adp_func']:
                keys = np.concatenate((np.array([-1]), cam_dict['keys']))

                # 1. find confident fg
                fg_conf_cam = np.pad(cam_dict['high_res'], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
                fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
                pred = imutils.crf_inference_label(img, fg_conf_cam, args.dataset, n_labels=keys.shape[0])
                fg_conf = keys[pred]

                # 2. combine confident fg & bg
                conf = fg_conf.copy()
                conf[fg_conf == -1] = 255
            elif args.dataset == 'voc12':
                keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

                # 1. find confident fg & bg
                fg_conf_cam = np.pad(cam_dict['high_res'], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
                fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
                pred = imutils.crf_inference_label(img, fg_conf_cam, args.dataset, n_labels=keys.shape[0])
                fg_conf = keys[pred]

                bg_conf_cam = np.pad(cam_dict['high_res'], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
                bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
                pred = imutils.crf_inference_label(img, bg_conf_cam, args.dataset, n_labels=keys.shape[0])
                bg_conf = keys[pred]

                # 2. combine confident fg & bg
                conf = fg_conf.copy()
                conf[fg_conf == 0] = 255
                conf[bg_conf + fg_conf == 0] = 0
            elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
                keys = np.concatenate((np.array([-1]), cam_dict['keys']))
                img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4))

                # 1. find confident fg
                fg_conf_cam = np.pad(cam_dict['cam'], ((1, 0), (0, 0), (0, 0)), mode='constant',
                                     constant_values=args.conf_fg_thres)
                fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
                pred = imutils.crf_inference_label(img, fg_conf_cam, args.dataset, n_labels=keys.shape[0])
                fg_conf = keys[pred]

                # 2. combine confident fg & bg
                conf = fg_conf.copy()
                conf[fg_conf == -1] = 255
            else:
                raise KeyError('Dataset %s not yet implemented' % args.dataset)

            imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'),
                            conf.astype(np.uint8))

            # Save with colour
            cam_clr = np.zeros(list(conf.shape) + [3], dtype=np.uint8)
            off = 0
            for t in ['bg', 'fg']:
                for i, c in enumerate(args.class_colours[t]):
                    for ch in range(3):
                        cam_clr[:, :, ch] += c[ch] * np.uint8(conf == (i + off))
                off += len(args.class_colours[t])
            for ch in range(3):
                cam_clr[:, :, ch] += 255 * np.uint8(conf == 255)
            imageio.imsave(os.path.join(args.ir_label_clr_out_dir, img_name + '.png'), cam_clr)
            # Save with colour, overlaid on original image
            cam_clr_over = np.uint8((1 - args.overlay_r) * np.float32(img) +
                                    args.overlay_r * np.float32(cam_clr))
            imageio.imsave(os.path.join(args.ir_label_clr_out_dir, img_name + '_overlay.png'), cam_clr_over)

            pbar.update(1)

        # if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
        #     print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

def run(args):
    if args.dataset == 'voc12':
        dataset = voc12.dataloader.VOC12ImageDataset(args.train_list, dev_root=args.dev_root, norm_mode=None,
                                                     to_torch=False)
    elif args.dataset in ['adp_morph', 'adp_func']:
        dataset = adp.dataloader.ADPImageDataset(args.train_list, dev_root=args.dev_root,
                                                 htt_type=args.dataset.split('_')[-1],
                                                 is_eval=args.split == 'evaluation', norm_mode=None, to_torch=False)
    elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
        dataset = deepglobe.dataloader.DeepGlobeImageDataset(args.train_list, dev_root=args.dev_root,
                                                             is_balanced=args.dataset == 'deepglobe_balanced',
                                                             norm_mode=None, to_torch=False)
    else:
        raise KeyError('Dataset %s not yet implemented' % args.dataset)
    dataset = torchutils.split_dataset(dataset, args.num_workers)

    # print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    # print(']')
