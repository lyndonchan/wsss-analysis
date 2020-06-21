
import torch
from torch.backends import cudnn
# from torch.utils.tensorboard import SummaryWriter
cudnn.enabled = True
from torch.utils.data import DataLoader
import os
import voc12.dataloader
import adp.dataloader
import deepglobe.dataloader
from misc import pyutils, torchutils, indexing
import importlib

def run(args):

    path_index = indexing.PathIndex(radius=10, default_size=(args.irn_crop_size // 4, args.irn_crop_size // 4))

    model = getattr(importlib.import_module(args.irn_network), 'AffinityDisplacementLoss')(path_index, args.model_dir,
                                                                                           args.dataset, args.tag,
                                                                                           args.num_classes,
                                                                                           args.use_cls)
    if args.dataset == 'voc12':
        train_dataset = voc12.dataloader.VOC12AffinityDataset(args.train_list,
                                                              label_dir=args.ir_label_out_dir,
                                                              dev_root=args.dev_root,
                                                              indices_from=path_index.src_indices,
                                                              indices_to=path_index.dst_indices,
                                                              hor_flip=True,
                                                              crop_size=args.irn_crop_size,
                                                              crop_method=args.crop_method,
                                                              rescale=args.rescale_range,
                                                              outsize=args.outsize,
                                                              norm_mode=args.norm_mode
                                                              )
        infer_dataset = voc12.dataloader.VOC12ImageDataset(args.infer_list,
                                                           dev_root=args.dev_root,
                                                           crop_size=args.irn_crop_size,
                                                           crop_method="top_left")
    elif args.dataset in ['adp_morph', 'adp_func']:
        train_dataset = adp.dataloader.ADPAffinityDataset(args.train_list,
                                                          is_eval=args.dataset == 'evaluation',
                                                          label_dir=args.ir_label_out_dir,
                                                          dev_root=args.dev_root,
                                                          htt_type=args.dataset.split('_')[-1],
                                                          indices_from=path_index.src_indices,
                                                          indices_to=path_index.dst_indices,
                                                          hor_flip=True,
                                                          crop_size=args.irn_crop_size,
                                                          crop_method=args.crop_method,
                                                          rescale=args.rescale_range,
                                                          outsize=args.outsize,
                                                          norm_mode=args.norm_mode
                                                          )
        infer_dataset = adp.dataloader.ADPImageDataset(args.infer_list,
                                                       dev_root=args.dev_root, htt_type=args.dataset.split('_')[-1],
                                                       is_eval=args.dataset == 'evaluation',
                                                       crop_size=args.irn_crop_size,
                                                       crop_method="top_left")
    elif args.dataset in ['deepglobe', 'deepglobe_balanced']:
        train_dataset = deepglobe.dataloader.DeepGlobeAffinityDataset(args.train_list,
                                                          is_balanced=args.dataset == 'deepglobe_balanced',
                                                          label_dir=args.ir_label_out_dir,
                                                          dev_root=args.dev_root,
                                                          indices_from=path_index.src_indices,
                                                          indices_to=path_index.dst_indices,
                                                          hor_flip=True,
                                                          crop_size=args.irn_crop_size,
                                                          crop_method=args.crop_method,
                                                          rescale=args.rescale_range,
                                                          outsize=args.outsize,
                                                          norm_mode=args.norm_mode
                                                          )
        infer_dataset = deepglobe.dataloader.DeepGlobeImageDataset(args.infer_list,
                                                       dev_root=args.dev_root,
                                                       is_balanced=args.dataset == 'deepglobe_balanced',
                                                       crop_size=args.irn_crop_size,
                                                       crop_method="top_left")
    else:
        raise KeyError('Dataset %s not yet implemented' % args.dataset)

    train_data_loader = DataLoader(train_dataset, batch_size=args.irn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.irn_batch_size) * args.irn_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay}
    ], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # writer = SummaryWriter('log_tb/' + args.run_name)

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.irn_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.irn_num_epoches))

        for iter, pack in enumerate(train_data_loader):

            img = pack['img'].cuda(non_blocking=True)
            bg_pos_label = pack['aff_bg_pos_label'].cuda(non_blocking=True)
            fg_pos_label = pack['aff_fg_pos_label'].cuda(non_blocking=True)
            neg_label = pack['aff_neg_label'].cuda(non_blocking=True)

            pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = model(img, True)

            bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
            fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
            pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
            neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

            dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
            dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

            avg_meter.add({'loss1': pos_aff_loss.item(), 'loss2': neg_aff_loss.item(),
                           'loss3': dp_fg_loss.item(), 'loss4': dp_bg_loss.item()})

            total_loss = (pos_aff_loss + neg_aff_loss) / 2 + (dp_fg_loss + dp_bg_loss) / 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                losses = {}
                for i in range(1,5):
                    losses[str(i)] = avg_meter.pop('loss'+str(i))

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % (losses['1'], losses['2'], losses['3'], losses['4']),
                      'imps:%.1f' % ((iter + 1) * args.irn_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                # writer.add_scalar('step', optimizer.global_step, ep * len(train_data_loader) + iter)
                # writer.add_scalar('loss', losses['1']+losses['2']+losses['3']+losses['4'],
                #                   ep * len(train_data_loader) + iter)
                # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep * len(train_data_loader) + iter)
        else:
            timer.reset_stage()
    infer_data_loader = DataLoader(infer_dataset, batch_size=args.irn_batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model.eval()
    print('Analyzing displacements mean ... ', end='')

    dp_mean_list = []

    with torch.no_grad():
        for iter, pack in enumerate(infer_data_loader):
            img = pack['img'].cuda(non_blocking=True)

            aff, dp = model(img, False)

            dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())

        model.module.mean_shift.running_mean = torch.mean(torch.stack(dp_mean_list), dim=0)
    print('done.')

    torch.save(model.module.state_dict(), args.irn_weights_name)
    torch.cuda.empty_cache()
