import os
from datetime import datetime
import configparser
import re
import func_sample

# Fixed settings
CFG = {}
CFG['adp_morph'] = {'model_ids': ['vgg16', 'x1.7'], 'train_set': 'train', 'val_set': 'tuning',
                    'add_sets': ['evaluation'], 'conf_fg_thres': [5, 5], 'init_exp_times': [2, 1]}
CFG['adp_func'] = {'model_ids': ['vgg16', 'x1.7'], 'train_set': 'train', 'val_set': 'tuning',
                   'add_sets': ['evaluation'], 'conf_fg_thres': [7, 3], 'init_exp_times': [3, 1]}
CFG['voc12'] = {'model_ids': ['vgg16', 'm7'], 'train_set': 'train_aug', 'val_set': 'val', 'add_sets': [],
                'conf_fg_thres': [5, 7], 'init_exp_times': [8, 3]}
CFG['deepglobe'] = {'model_ids': ['vgg16', 'm7'], 'train_set': 'train75', 'val_set': 'test', 'add_sets': [],
                    'conf_fg_thres': [5, 5], 'init_exp_times': [4, 8]}
CFG['deepglobe_balanced'] = {'model_ids': ['vgg16', 'm7'], 'train_set': 'train37.5', 'val_set': 'test', 'add_sets': [],
                             'conf_fg_thres': [4, 7], 'init_exp_times': [7, 7]}

# Variable settings
config = configparser.ConfigParser()
config.read('../settings.ini')
DATA_ROOT = config['Download Directory']['data_dir']
MODEL_DIR = os.path.join(config['Download Directory']['data_dir'], config['Data Folders']['model_cnn_dir'])
IRN_BATCH_SIZE = 4
DATASETS = ['adp_morph', 'adp_func', 'voc12', 'deepglobe', 'deepglobe_balanced']
assert all([d in CFG for d in DATASETS])

def read_miou_from_log(run_name):
    log_path = run_name + '.log'
    mious = []
    with open(log_path, 'r') as f:
        for line in f.readlines():
            if re.match(r'\[eval_cam, [a-z]+\] miou: [0-9]+', line.rstrip()):
                mious.append(float(line.split('miou: ')[-1]))
    return mious[-1]

if __name__ == '__main__':
    for dataset in DATASETS:
        for model_id, curr_thres, curr_exp in zip(CFG[dataset]['model_ids'], CFG[dataset]['conf_fg_thres'],
                                                  CFG[dataset]['init_exp_times']):
            run_name = '%s_%s_cam' % (dataset, model_id)
            # Evaluate on validation set
            val_split = CFG[dataset]['val_set']
            func_sample.sample(MODEL_DIR, model_id, dataset, val_split, IRN_BATCH_SIZE, curr_thres, curr_exp,
                               run_name, '--make_cam_pass', '--eval_cam_pass', '--make_sem_seg_pass',
                               '--eval_sem_seg_pass')
            # Evaluate on additional sets (if applicable)
            if len(CFG[dataset]['add_sets']) > 0:
                for add_set in CFG[dataset]['add_sets']:
                    func_sample.sample(MODEL_DIR, model_id, dataset, add_set, IRN_BATCH_SIZE, curr_thres, curr_exp,
                                       run_name, '--make_cam_pass', '--eval_cam_pass', '--make_sem_seg_pass',
                                       '--eval_sem_seg_pass')