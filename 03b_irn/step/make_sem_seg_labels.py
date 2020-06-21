import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import voc12.dataloader
import adp.dataloader
import deepglobe.dataloader
from misc import torchutils, indexing

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    # data_loader = DataLoader(databin,
    #                          shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    data_loader = DataLoader(databin, shuffle=False, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader)):
            # img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            img_name = pack['name'][0]
            orig_img_size = np.asarray(pack['size'])

            if args.dataset == 'voc12':
                img_path = voc12.dataloader.get_img_path(img_name, args.dev_root)
            elif args.dataset in ['adp_morph', 'adp_func']:
                img_path = adp.dataloader.get_img_path(img_name, args.dev_root, args.split == 'evaluation')
            elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
                img_path = deepglobe.dataloader.get_img_path(img_name, args.dev_root)
            else:
                raise KeyError('Dataset %s not yet implemented' % args.dataset)

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))
            # if img_name == '2007_001185':
            #     cv2.imwrite('edge.png', np.uint8(255 * cv2.resize(edge.cpu().numpy()[0], tuple(orig_img_size[::-1]))))
            #     D = dp.cpu().numpy()
            #     hsv = np.zeros((D.shape[1], D.shape[2], 3), dtype='uint8')
            #     hsv[..., 1] = 255
            #     mag, ang = cv2.cartToPolar(-D[0], -D[1])
            #     hsv[..., 0] = ang * 180 / np.pi / 2
            #     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #
            #     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #     cv2.imwrite('dp.png', cv2.resize(rgb[:, :, ::-1], tuple(orig_img_size[::-1])))
            #     a=1
            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cam_downsized_values = torch.from_numpy(cam_dict['cam']).cuda()
            if args.dataset == 'voc12':
                if len(cam_dict['keys']) > 0:
                    keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                    if edge.shape[1:] != cam_downsized_values.shape[1:]:
                        edge = F.interpolate(edge.unsqueeze(0), size=(cam_downsized_values.shape[1:]), mode='bilinear', align_corners=False)

                    rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5) # radius=5
                    # rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=3, radius=5)

                    rw_up = F.interpolate(rw, size=tuple(orig_img_size), mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
                    rw_up = rw_up / torch.max(rw_up)

                    rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
                    rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

                    rw_pred = keys[rw_pred]
                else:
                    rw_pred = np.zeros(orig_img_size, dtype='uint8')
            elif args.dataset in ['adp_morph', 'adp_func']:
                keys = cam_dict['keys']

                if edge.shape[1:] != cam_downsized_values.shape[1:]:
                    edge = F.interpolate(edge.unsqueeze(0), size=(cam_downsized_values.shape[1:]), mode='bilinear',
                                         align_corners=False)

                rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times,
                                                radius=5)
                rw_up = F.interpolate(rw, size=tuple(orig_img_size), mode='bilinear', align_corners=False)[..., 0,
                        :orig_img_size[0], :orig_img_size[1]]
                rw_up = rw_up / torch.max(rw_up)
                rw_pred = torch.argmax(rw_up, dim=0).cpu().numpy()

                rw_pred = keys[rw_pred]
            elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
                if len(cam_dict['keys']) > 0:
                    keys = cam_dict['keys']

                    down_fac = 6
                    cam_downsized_values = F.interpolate(cam_downsized_values.unsqueeze(0),
                                      size=[x // down_fac for x in cam_downsized_values.shape[1:]],
                                      mode='bilinear', align_corners=False)[0]
                    edge = F.interpolate(edge.unsqueeze(0), size=(cam_downsized_values.shape[1:]), mode='bilinear',
                                         align_corners=False)

                    rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times,
                                                    radius=5)
                    rw_up = F.interpolate(rw, size=tuple(orig_img_size // 4), mode='bilinear', align_corners=False)[..., 0,
                            :orig_img_size[0] // 4, :orig_img_size[1] // 4]
                    rw_up = rw_up / torch.max(rw_up)
                    rw_pred = torch.argmax(rw_up, dim=0).cpu().numpy()

                    rw_pred = keys[rw_pred]
                else:
                    rw_pred = 5 * np.ones(tuple(orig_img_size // 4))
            else:
                raise KeyError('Dataset %s not yet implemented' % args.dataset)

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            # Save with colour
            rw_pred_clr = np.zeros(list(rw_pred.shape) + [3], dtype=np.uint8)
            off = 0
            for t in ['bg', 'fg']:
                for i, c in enumerate(args.class_colours[t]):
                    for ch in range(3):
                        rw_pred_clr[:, :, ch] += c[ch] * np.uint8(rw_pred == (i + off))
                off += len(args.class_colours[t])
            imageio.imsave(os.path.join(args.sem_seg_clr_out_dir, img_name + '.png'), rw_pred_clr)
            # Save with colour, overlaid on original image
            if args.dataset not in ['deepglobe', 'deepglobe_balanced']:
                orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            else:
                orig_img = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), rw_pred_clr.shape[:2])
            if args.dataset in ['adp_morph', 'adp_func']:
                rw_pred_clr = cv2.resize(rw_pred_clr, orig_img.shape[:2])
            rw_pred_clr_over = np.uint8((1 - args.overlay_r) * np.float32(orig_img) +
                                 args.overlay_r * np.float32(rw_pred_clr))
            imageio.imsave(os.path.join(args.sem_seg_clr_out_dir, img_name + '_overlay.png'), rw_pred_clr_over)

            # if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
            #     print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')(args.model_dir, args.dataset,
                                                                                   args.tag, args.num_classes,
                                                                                   args.use_cls)
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    if args.dataset == 'voc12':
        dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list, norm_mode=args.norm_mode,
                                                                 outsize=args.outsize, dev_root=args.dev_root,
                                                                 scales=(1.0,))
    elif args.dataset in ['adp_morph', 'adp_func']:
        dataset = adp.dataloader.ADPClassificationDatasetMSF(args.infer_list, norm_mode=args.norm_mode,
                                                             outsize=args.outsize, dev_root=args.dev_root,
                                                             htt_type=args.dataset.split('_')[-1],
                                                             is_eval=args.split == 'evaluation', scales=(1.0,))
    elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
        dataset = deepglobe.dataloader.DeepGlobeClassificationDatasetMSF(args.infer_list, norm_mode=args.norm_mode,
                                                             outsize=args.outsize, dev_root=args.dev_root,
                                                             is_balanced=args.dataset == 'deepglobe_balanced',
                                                             scales=(1.0,))
    else:
        raise KeyError('Dataset %s not yet implemented' % args.dataset)

    dataset = torchutils.split_dataset(dataset, n_gpus)

    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)

    torch.cuda.empty_cache()
