import argparse
import os
import configparser
import numpy as np
import sys

from misc import pyutils

config = configparser.ConfigParser()
config.read('../settings.ini')
DATA_ROOT = config['Download Directory']['data_dir']
MODEL_WSSS_ROOT = os.path.join(config['Download Directory']['data_dir'], config['Data Folders']['model_wsss_dir'])

def sample(*args):
    model_dir, model_id, dataset, split, irn_batch_size, conf_fg_thres, exp_times, run_name, *args_res = args
    _run(['--model_dir', model_dir, '--model_id', model_id, '--dataset', dataset, '--split', split,
          '--irn_batch_size', str(irn_batch_size), '--conf_fg_thres', str(conf_fg_thres), '--exp_times', str(exp_times),
          '--run_name', run_name] + args_res)

def get_cmd_str(cmd):
    cmd_str = ''
    for i, x in enumerate(cmd):
        if i == 0:
            cmd_str += x
        elif i > 0 and cmd[i - 1].startswith('--') and not x.startswith('--'):
            cmd_str += '=' + x
        else:
            cmd_str += ' ' + x
    return cmd_str

def _run(cmd):
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--root", required=False, default=DATA_ROOT, type=str,
                        help="Path to root directory containing development kits as subdirectories.")
    parser.add_argument("--model_dir", type=str,
                        help="Path to model directory, must contain .json architecture and .h5 weights files")
    parser.add_argument("--model_id", type=str, choices=['resnet50', 'vgg16', 'm7', 'x1.7'],
                        help="Identifier of the model to run: resnet50 (legacy), vgg16, m7, x1.7")
    parser.add_argument("--run_name", type=str, default='', help="Name to save the relevant results/models to in file")

    # Dataset
    parser.add_argument("--dataset", type=str, choices=['voc12', 'adp_morph', 'adp_func', 'deepglobe', 'deepglobe_balanced'],
                        help="Name of the dataset to run on: voc12, adp_morph, adp_func, deepglobe, deepglobe_balanced")
    parser.add_argument("--split", type=str, help="Name of the dataset split to run on: train_aug, val")

    # Class Activation Map
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=5e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0,), # default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8, type=int,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Step
    parser.add_argument('--train_cam_pass', default=False, dest='train_cam_pass', action='store_true')
    parser.add_argument("--make_cam_pass", default=False, dest='make_cam_pass', action='store_true')
    parser.add_argument("--eval_cam_pass", default=False, dest='eval_cam_pass', action='store_true')
    parser.add_argument("--cam_to_ir_label_pass", default=False, dest='cam_to_ir_label_pass', action='store_true')
    parser.add_argument("--train_irn_pass", default=False, dest='train_irn_pass', action='store_true')
    parser.add_argument("--make_sem_seg_pass", default=False, dest='make_sem_seg_pass', action='store_true')
    parser.add_argument("--eval_sem_seg_pass", default=False, dest='eval_sem_seg_pass', action='store_true')

    args = parser.parse_args(cmd)

    if len(args.run_name) == 0:
        args.run_name = args.dataset + '_' + args.model_id + '_t' + str(10*args.conf_fg_thres) + '_e' + \
                        str(args.exp_times)
    if args.dataset == 'voc12':
        assert args.split in ['train_aug', 'val']
        if args.model_id == 'vgg16':
            args.tag = 'VOC2012_VGG16'
        elif args.model_id == 'm7':
            args.tag = 'VOC2012_M7'
        args.dev_root = os.path.join(args.root, 'VOCdevkit', 'VOC2012')
    elif args.dataset in ['adp_morph', 'adp_func']:
        assert args.split in ['train', 'tuning', 'evaluation']
        if args.model_id == 'vgg16':
            args.tag = 'ADP_VGG16'
        elif args.model_id == 'x1.7':
            args.tag = 'ADP_X1.7'
        args.dev_root = os.path.join(args.root, 'ADPdevkit', 'ADPRelease1')
    elif args.dataset == 'deepglobe':
        assert args.split in ['train75', 'test']
        if args.model_id == 'vgg16':
            args.tag = 'DeepGlobe_train75_VGG16'
        elif args.model_id == 'm7':
            args.tag = 'DeepGlobe_train75_M7'
        args.dev_root = os.path.join(args.root, 'DGdevkit')
    elif args.dataset == 'deepglobe_balanced':
        assert args.split in ['train37.5', 'test']
        if args.model_id == 'vgg16':
            args.tag = 'DeepGlobe_train37.5_VGG16'
        elif args.model_id == 'm7':
            args.tag = 'DeepGlobe_train37.5_M7'
        args.dev_root = os.path.join(args.root, 'DGdevkit')
    if args.dataset in ['adp_morph', 'adp_func']:
        args.train_list = os.path.join('adp', args.split + '.txt')
        args.val_list = os.path.join('adp', args.split + '.txt')
        args.infer_list = os.path.join('adp', args.split + '.txt')
    elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
        args.train_list = os.path.join('deepglobe', args.split + '.txt')
        args.val_list = os.path.join('deepglobe', args.split + '.txt')
        args.infer_list = os.path.join('deepglobe', args.split + '.txt')
    else:
        args.train_list = os.path.join(args.dataset, args.split + '.txt')
        args.val_list = os.path.join(args.dataset, args.split + '.txt')
        args.infer_list = os.path.join(args.dataset, args.split + '.txt')
    args.chainer_eval_set = os.path.join(args.split)

    if args.model_id == 'vgg16':
        args.in_img_sz = 321
        args.outsize = (321, 321)
        args.norm_mode = 'int'
        args.crop_method = None
        args.rescale_range = None
    elif args.model_id in ['m7', 'x1.7']:
        args.in_img_sz = 224
        args.outsize = (224, 224)
        args.norm_mode = 'int'
        args.crop_method = None
        args.rescale_range = None
    elif args.model_id == 'resnet50':
        args.in_img_sz = 512
        args.outsize = None
        args.norm_mode = 'int'
        args.crop_method = 'random'
        args.rescale_range = (0.5, 1.5)

    args.class_names = {}
    args.class_colours = {}
    if args.dataset == 'voc12':
        args.class_names['fg'] = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                                  'train', 'tvmonitor']
        args.class_names['bg'] = ['background']
        args.class_colours['fg'] = np.array([(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
                                             (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                                             (0, 64, 128)])
        args.class_colours['bg'] = np.array([(0, 0, 0)])
        args.overlay_r = 0.75
        args.num_classes = len(args.class_names['fg'])
        args.use_cls = list(range(args.num_classes))
    elif args.dataset == 'adp_morph':
        args.class_names['fg'] = ['E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I', 'C.D.R', 'C.L',
                                  'H.E', 'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W', 'A.B', 'A.M',
                                  'M.M', 'M.K', 'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W']
        args.class_names['bg'] = ['background']
        args.class_colours['fg'] = np.array([(0, 0, 128), (0, 128, 0), (255, 165, 0), (255, 192, 203), (255, 0, 0),
                                             (173, 20, 87), (176, 141, 105), (3, 155, 229), (158, 105, 175),
                                             (216, 27, 96), (244, 81, 30), (124, 179, 66), (142, 36, 255), (240, 147, 0),
                                             (204, 25, 165), (121, 85, 72), (142, 36, 170), (179, 157, 219), (121, 134, 203),
                                             (97, 97, 97), (167, 155, 142), (228, 196, 136), (213, 0, 0), (4, 58, 236),
                                             (0, 150, 136), (228, 196, 65), (239, 108, 0), (74, 21, 209)])
        args.class_colours['bg'] = np.array([(255, 255, 255)])
        args.overlay_r = 0.75
        args.num_classes = 31
        args.use_cls = list(range(args.num_classes-3))
    elif args.dataset == 'adp_func':
        args.class_names['fg'] = ['G.O', 'G.N', 'T']
        args.class_names['bg'] = ['background', 'other']
        args.class_colours['fg'] = np.array([(0, 0, 128), (0, 128, 0), (173, 20, 87)])
        args.class_colours['bg'] = np.array([(255, 255, 255), (3, 155, 229)])
        args.overlay_r = 0.75
        args.num_classes = 31
        args.use_cls = list(range(args.num_classes-3, args.num_classes))
    elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
        args.class_names['fg'] = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren', 'unknown']
        args.class_names['bg'] = []
        args.class_colours['fg'] = np.array([(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255),
                                       (255, 255, 255), (0, 0, 0)])
        args.class_colours['bg'] = np.empty(0)
        args.overlay_r = 0.25
        args.num_classes = len(args.class_colours['fg'])
        args.use_cls = [0, 1, 2, 3, 4, 5]
    if args.model_id != 'x1.7':
        args.cam_network = 'net.' + args.model_id + '_cam'
    else:
        args.cam_network = 'net.m7_cam'
    args.cam_crop_size = args.in_img_sz
    if args.model_id != 'x1.7':
        args.irn_network = 'net.' + args.model_id + '_irn'
    else:
        args.irn_network = 'net.m7_irn'
    args.irn_crop_size = args.in_img_sz
    args.cam_weights_name = os.path.join(MODEL_WSSS_ROOT, 'IRNet', args.dataset + '_' + args.model_id + '_cam.pth')
    irn_name = args.dataset + '_' + args.model_id + '_t' + str(10*args.conf_fg_thres)
    args.irn_weights_name = os.path.join(MODEL_WSSS_ROOT, 'IRNet', irn_name + '_irn.pth')
    args.cam_out_dir = os.path.join('out', args.run_name, 'cam_' + args.split)
    args.cam_clr_out_dir = os.path.join('out', args.run_name, 'cam_clr_' + args.split)
    args.ir_label_out_dir = os.path.join('out', args.run_name, 'ir_label_' + args.split)
    args.ir_label_clr_out_dir = os.path.join('out', args.run_name, 'ir_label_clr_' + args.split)
    args.sem_seg_out_dir = os.path.join('out', args.run_name, 'sem_seg_' + args.split)
    args.sem_seg_clr_out_dir = os.path.join('out', args.run_name, 'sem_seg_clr_' + args.split)
    args.eval_dir = os.path.join('eval', args.run_name + '_' + args.split)
    args.log_dir = 'log'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(MODEL_WSSS_ROOT, 'IRNet'), exist_ok=True)

    args.logfile = os.path.join(args.log_dir, args.run_name + '.log')
    cmd_str = get_cmd_str(cmd)
    with open(args.logfile, 'a') as f:
        f.write(cmd_str + '\n')

    for k in sorted(vars(args)):
        print('*', k, ':', vars(args)[k])
    # print(vars(args))

    if args.make_cam_pass is True:
        import step.make_cam
        os.makedirs(args.cam_out_dir, exist_ok=True)

        print('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam
        os.makedirs(args.eval_dir, exist_ok=True)
        os.makedirs(args.cam_clr_out_dir, exist_ok=True)

        print('step.eval_cam:')
        step.eval_cam.run(args)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label
        os.makedirs(args.ir_label_out_dir, exist_ok=True)
        os.makedirs(args.ir_label_clr_out_dir, exist_ok=True)

        print('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        print('step.train_irn:')
        step.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels
        os.makedirs(args.sem_seg_out_dir, exist_ok=True)
        os.makedirs(args.sem_seg_clr_out_dir, exist_ok=True)

        print('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg
        os.makedirs(args.eval_dir, exist_ok=True)

        print('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)

