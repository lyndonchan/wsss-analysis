import os
import configparser
from datetime import datetime
import re
import func_sample

# File paths
config = configparser.ConfigParser()
config.read('../settings.ini')
MODEL_DIR = os.path.join(config['Download Directory']['data_dir'], config['Data Folders']['model_cnn_dir'])

# Fixed settings
CFG = {}
CFG['adp_morph'] = {'model_ids': ['vgg16', 'x1.7'], 'train_set': 'train', 'val_set': 'tuning',
                    'add_sets': ['evaluation'], 'init_exp_times': [1, 1]}
CFG['adp_func'] = {'model_ids': ['vgg16', 'x1.7'], 'train_set': 'train', 'val_set': 'tuning',
                   'add_sets': ['evaluation'], 'init_exp_times': [1, 1]}
CFG['voc12'] = {'model_ids': ['vgg16', 'm7'], 'train_set': 'train_aug', 'val_set': 'val', 'add_sets': [],
                'init_exp_times': [8, 3]}
CFG['deepglobe'] = {'model_ids': ['vgg16', 'm7'], 'train_set': 'train75', 'val_set': 'test', 'add_sets': [],
                    'init_exp_times': [1, 1]}
CFG['deepglobe_balanced'] = {'model_ids': ['vgg16', 'm7'], 'train_set': 'train37.5', 'val_set': 'test', 'add_sets': [],
                             'init_exp_times': [1, 1]}
THRES_RNG = [0.3, 0.5, 0.7]
EXP_RNG = [1, 2, 3, 4, 5, 6, 7, 8]

# Variable settings
IRN_BATCH_SIZE = 4
DATASETS = ['adp_func'] # ['adp_morph', 'adp_func', 'voc12', 'deepglobe', 'deepglobe_balanced']
assert all([d in CFG for d in DATASETS])

def read_miou_from_log(run_name):
    log_path = run_name + '.log'
    mious = []
    with open(log_path, 'r') as f:
        for line in f.readlines():
            if re.match(r'\[eval_sem_seg, [a-z]+\] miou: [0-9]+', line.rstrip()):
                mious.append(float(line.split('miou: ')[-1]))
    return mious[-1]

def write_to_tuning_log(tuning_log_id, s):
    with open('tuning_logs/tuning_log_' + tuning_log_id + '.log', 'a') as f:
        f.write(s + '\n')

if __name__ == '__main__':

    tuning_log_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('tuning_logs'):
        os.makedirs('tuning_logs')
    write_to_tuning_log(tuning_log_id, 'dataset\tmodel\tconf_fg_thres\texp_times\tvalidation miou')
    for dataset in DATASETS:
        for model_id, init_exp in zip(CFG[dataset]['model_ids'], CFG[dataset]['init_exp_times']):
            curr_exp = init_exp
            # Find best threshold
            miou_opt_thres = []
            for curr_thres in THRES_RNG:
                run_name = '%s_%s_t%de%d' % (dataset, model_id, int(curr_thres*10), curr_exp)
                # Train on training set
                train_split = CFG[dataset]['train_set']
                func_sample.sample(MODEL_DIR, model_id, dataset, train_split, IRN_BATCH_SIZE, curr_thres, curr_exp,
                                   run_name, '--make_cam_pass', '--cam_to_ir_label_pass', '--train_irn_pass')
                # Evaluate on validation set
                val_split = CFG[dataset]['val_set']
                func_sample.sample(MODEL_DIR, model_id, dataset, val_split, IRN_BATCH_SIZE, curr_thres, curr_exp,
                                   run_name, '--make_cam_pass', '--eval_cam_pass', '--make_sem_seg_pass',
                                   '--eval_sem_seg_pass')
                miou_opt_thres.append(read_miou_from_log('log/' + run_name))
                write_to_tuning_log(tuning_log_id, '%s\t%s\t%.1f\t%d\t%f' % (dataset, model_id, curr_thres, curr_exp, miou_opt_thres[-1]))

                # Evaluate on additional sets (if applicable)
                if len(CFG[dataset]['add_sets']) > 0:
                    for add_set in CFG[dataset]['add_sets']:
                        func_sample.sample(MODEL_DIR, model_id, dataset, add_set, IRN_BATCH_SIZE, curr_thres, curr_exp,
                                           run_name, '--make_cam_pass', '--eval_cam_pass', '--make_sem_seg_pass',
                                           '--eval_sem_seg_pass')
            opt_thres = THRES_RNG[miou_opt_thres.index(max(miou_opt_thres))]
            # Find best exp_times
            miou_opt_exp = []
            for curr_exp in [x for x in EXP_RNG if x != init_exp]:
                run_name = '%s_%s_t%de%d' % (dataset, model_id, int(opt_thres*10), curr_exp)
                # Evaluate on validation set
                val_split = CFG[dataset]['val_set']
                func_sample.sample(MODEL_DIR, model_id, dataset, val_split, IRN_BATCH_SIZE, opt_thres, curr_exp,
                                   run_name, '--make_cam_pass', '--eval_cam_pass', '--make_sem_seg_pass',
                                   '--eval_sem_seg_pass')
                miou_opt_exp.append(read_miou_from_log('log/' + run_name))
                write_to_tuning_log(tuning_log_id, '%s\t%s\t%.1f\t%d\t%f' % (dataset, model_id, opt_thres, curr_exp,
                                                                             miou_opt_exp[-1]))
                # Evaluate on additional sets (if applicable)
                if len(CFG[dataset]['add_sets']) > 0:
                    for add_set in CFG[dataset]['add_sets']:
                        func_sample.sample(MODEL_DIR, model_id, dataset, add_set, IRN_BATCH_SIZE, opt_thres, curr_exp,
                                           run_name, '--make_cam_pass', '--eval_cam_pass', '--make_sem_seg_pass',
                                           '--eval_sem_seg_pass')
            opt_exp = EXP_RNG[miou_opt_exp.index(max(miou_opt_exp))]