import argparse
from deepglobe import dataloader
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default='train37.5.txt', type=str)
    parser.add_argument("--eval_list", default='test.txt', type=str)
    parser.add_argument("--out", default="cls_labels_balanced.npy", type=str)
    parser.add_argument("--root", default="../../database/DGdevkit", type=str)
    args = parser.parse_args()

    train_name_list = dataloader.load_img_name_list(args.train_list)
    eval_name_list = dataloader.load_img_name_list(args.eval_list)

    train_eval_name_list = np.concatenate([train_name_list, eval_name_list], axis=0)
    train_label_list = dataloader.load_image_label_list_from_csv(train_name_list, args.root, 'train37.5.csv')
    eval_label_list = dataloader.load_image_label_list_from_csv(eval_name_list, args.root, 'test.csv')
    label_list = np.concatenate([train_label_list, eval_label_list], axis=0)

    total_label = np.zeros(len(dataloader.CAT_LIST))

    d = dict()
    for img_name, label in zip(train_eval_name_list, label_list):
        d[img_name] = label
        total_label += label

    print(total_label)
    np.save(args.out, d)