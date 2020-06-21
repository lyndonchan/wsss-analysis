import argparse
from adp import dataloader
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default='train.txt', type=str)
    parser.add_argument("--tuning_list", default='tuning.txt', type=str)
    parser.add_argument("--eval_list", default='evaluation.txt', type=str)
    parser.add_argument("--out", default="cls_labels_func.npy", type=str)
    parser.add_argument("--root", default="../../database/ADPdevkit/ADPRelease1", type=str)
    parser.add_argument("--htt_type", default="func", type=str)
    args = parser.parse_args()

    train_name_list = dataloader.load_img_name_list(args.train_list)
    tuning_name_list = dataloader.load_img_name_list(args.tuning_list)
    eval_name_list = dataloader.load_img_name_list(args.eval_list)

    train_val_name_list = np.concatenate([train_name_list, tuning_name_list, eval_name_list], axis=0)
    label_list = dataloader.load_image_label_list_from_csv(train_val_name_list, args.root, args.htt_type)

    total_label = np.zeros(len(dataloader.CAT_LIST[args.htt_type]))

    d = dict()
    for img_name, label in zip(train_val_name_list, label_list):
        d[img_name] = label
        total_label += label

    print(total_label)
    np.save(args.out, d)