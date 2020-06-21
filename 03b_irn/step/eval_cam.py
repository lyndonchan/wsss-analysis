
import numpy as np
import os
import cv2
from tqdm import tqdm
import imageio
import pandas as pd

import voc12.dataloader
import adp.dataloader
import deepglobe.dataloader

from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from adp.adp_semantic_segmentation_dataset import ADPSemanticSegmentationDataset
from deepglobe.deepglobe_semantic_segmentation_dataset import DeepGlobeSemanticSegmentationDataset
import matplotlib.pyplot as plt

def run(args):

    if args.dataset == 'voc12':
        dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.dev_root)
        outsize = None
    elif args.dataset in ['adp_morph', 'adp_func']:
        dataset = ADPSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.dev_root,
                                                 htt_type=args.dataset.split('_')[-1])
        outsize = (1088, 1088)
    elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
        dataset = DeepGlobeSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.dev_root,
                                                       is_balanced=args.dataset == 'deepglobe_balanced')
        outsize = (2448, 2448)
    else:
        raise KeyError('Dataset %s not yet implemented' % args.dataset)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    with tqdm(total=len(dataset)) as pbar:
        for id in dataset.ids:
            if args.dataset == 'voc12':
                img_path = voc12.dataloader.get_img_path(id, args.dev_root)
            elif args.dataset in ['adp_morph', 'adp_func']:
                img_path = adp.dataloader.get_img_path(id, args.dev_root, args.split == 'evaluation')
            elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
                img_path = deepglobe.dataloader.get_img_path(id, args.dev_root)
            else:
                raise KeyError('Dataset %s not yet implemented' % args.dataset)

            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            if args.dataset == 'voc12':
                cams = cam_dict['high_res']
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
                keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            elif args.dataset in ['adp_morph', 'adp_func']:
                keys = cam_dict['keys']
                cams = cam_dict['high_res']
            elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
                keys = cam_dict['keys']
                cams = cam_dict['cam']
            else:
                raise KeyError('Dataset %s not yet implemented' % args.dataset)
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            if outsize is not None:
                cls_labels = cv2.resize(cls_labels, outsize, interpolation=cv2.INTER_NEAREST)

            imageio.imsave(os.path.join(args.cam_clr_out_dir, id + '.png'), cls_labels.astype(np.uint8))
            # Save with colour
            rw_pred_clr = np.zeros(list(cls_labels.shape) + [3], dtype=np.uint8)
            off = 0
            for t in ['bg', 'fg']:
                for i, c in enumerate(args.class_colours[t]):
                    for ch in range(3):
                        rw_pred_clr[:, :, ch] += c[ch] * np.uint8(cls_labels == (i + off))
                off += len(args.class_colours[t])
            imageio.imsave(os.path.join(args.cam_clr_out_dir, id + '.png'), rw_pred_clr)
            # Save with colour, overlaid on original image
            if args.dataset not in ['deepglobe', 'deepglobe_balanced']:
                orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            else:
                orig_img = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), rw_pred_clr.shape[:2])
            if args.dataset in ['adp_morph', 'adp_func']:
                rw_pred_clr = cv2.resize(rw_pred_clr, orig_img.shape[:2])
            rw_pred_clr_over = np.uint8((1 - args.overlay_r) * np.float32(orig_img) +
                                        args.overlay_r * np.float32(rw_pred_clr))
            imageio.imsave(os.path.join(args.cam_clr_out_dir, id + '_overlay.png'), rw_pred_clr_over)
            preds.append(cls_labels.copy())
            pbar.update(1)

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    precision = gtjresj / resj
    recall = gtjresj / gtj
    miou = np.array([np.nanmean(iou)])
    mprecision = np.array([np.nanmean(precision)])
    mrecall = np.array([np.nanmean(recall)])

    iou_data = np.concatenate((iou, miou), axis=0)
    pr_data = np.concatenate((precision, mprecision), axis=0)
    re_data = np.concatenate((recall, mrecall), axis=0)
    data = np.column_stack((iou_data, pr_data, re_data))
    if args.dataset in ['deepglobe', 'deepglobe_balanced']:
        row_names = args.class_names['bg'] + args.class_names['fg'][:-1] + ['mean']
    else:
        row_names = args.class_names['bg'] + args.class_names['fg'] + ['mean']
    df = pd.DataFrame(data, index=row_names, columns=['iou', 'precision', 'recall'])
    df.to_csv(os.path.join(args.eval_dir, args.run_name + '_' + args.split + '_cam_iou.csv'), index=True)

    with open(args.logfile, 'a') as f:
        f.write('[eval_cam, ' + args.split + '] iou: ' + str(list(iou)) + '\n')
        f.write('[eval_cam, ' + args.split + '] miou: ' + str(miou[0]) + '\n')
    # args.logger.write('[eval_cam] iou: ' + iou + '\n')
    # args.logger.write('[eval_cam] miou: ' + miou+ '\n')