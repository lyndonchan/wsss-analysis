
import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from adp.adp_semantic_segmentation_dataset import ADPSemanticSegmentationDataset
from deepglobe.deepglobe_semantic_segmentation_dataset import DeepGlobeSemanticSegmentationDataset
import imageio

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
            cls_labels = imageio.imread(os.path.join(args.sem_seg_out_dir, id + '.png')).astype(np.uint8)
            cls_labels[cls_labels == 255] = 0
            if outsize is not None:
                cls_labels = cv2.resize(cls_labels, outsize, interpolation=cv2.INTER_NEAREST)
            preds.append(cls_labels.copy())
            pbar.update(1)

    confusion = calc_semantic_segmentation_confusion(preds, labels)#[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    miou = np.array([np.nanmean(iou)])

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    data = np.concatenate((iou, miou), axis=0)
    if args.dataset in ['deepglobe', 'deepglobe_balanced']:
        row_names = args.class_names['bg'] + args.class_names['fg'][:-1] + ['miou']
    else:
        row_names = args.class_names['bg'] + args.class_names['fg'] + ['miou']
    df = pd.DataFrame(data, index=row_names, columns=['iou'])
    df.to_csv(os.path.join(args.eval_dir, args.run_name + '_' + args.split + '_iou.csv'), index=True)

    with open(args.logfile, 'a') as f:
        f.write('[eval_sem_seg, ' + args.split + '] iou: ' + str(list(iou)) + '\n')
        f.write('[eval_sem_seg, ' + args.split + '] miou: ' + str(miou[0]) + '\n')