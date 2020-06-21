import os
import pandas as pd
import numpy as np

SETS = ['ADP-morph_tuning_VGG16', 'ADP-morph_tuning_X1.7', 'ADP-morph_segtest_VGG16', 'ADP-morph_segtest_X1.7',
        'ADP-func_tuning_VGG16', 'ADP-func_tuning_X1.7', 'ADP-func_segtest_VGG16', 'ADP-func_segtest_X1.7',
        'VOC2012_VGG16', 'VOC2012_M7',
        'DeepGlobe_VGG16', 'DeepGlobe_M7',
        'DeepGlobe_balanced_VGG16', 'DeepGlobe_balanced_M7']

# SEC/DSRG
def to_underscore(x):
    return x.replace('-VGG16', '_VGG16').replace('-X1.7', '_X1.7').replace('-M7', '_M7')

def to_dash(x):
    return x.replace('_VGG16', '-VGG16').replace('_X1.7', '-X1.7').replace('_M7', '-M7')

DIR = '../03a_sec-dsrg/eval'
eval_sec_dsrg = {'SEC': {}, 'DSRG': {}}
def get_miou(fpath):
    if not os.path.exists(fpath):
        return np.nan
    else:
        df = pd.read_excel(fpath)
        return df['IoU'][df['Class'] == 'Mean'].values[0]
for method in ['SEC', 'DSRG']:
    folders = os.listdir(os.path.join(DIR, method))
    for folder in folders:
        if 'ADP' in folder:
            for s in [to_dash(folder.replace('train', 'tuning')), to_dash(folder.replace('train', 'segtest'))]:
                fpath = os.path.join(DIR, method, folder, 'metrics_%s.xlsx' % s)
                key = to_underscore(s)
                eval_sec_dsrg[method][key] = get_miou(fpath)
        elif 'DeepGlobe' in folder:
            s = to_dash(folder.replace('train_', 'test_'))
            fpath = os.path.join(DIR, method, folder, 'metrics_%s.xlsx' % s)
            key = folder.replace('_train_', '_')
            eval_sec_dsrg[method][key] = get_miou(fpath)
        else:
            s = to_dash(folder.replace('train_', 'val_'))
            fpath = os.path.join(DIR, method, folder, 'metrics_%s.xlsx' % s)
            key = to_underscore(s).replace('val_', '')
            eval_sec_dsrg[method][key] = get_miou(fpath)

# Grad-CAM/IRNet
DIR = '../03b_irn/eval'
folders = os.listdir(DIR)
eval_cam = {}
eval_irn = {}
def irn_folder_to_key(folder):
    if folder.startswith('adp_morph'):
        key = 'ADP-morph'
    elif folder.startswith('adp_func'):
        key = 'ADP-func'
    elif folder.startswith('voc12'):
        key = 'VOC2012'
    elif folder.startswith('deepglobe_balanced'):
        key = 'DeepGlobe_balanced'
    elif folder.startswith('deepglobe'):
        key = 'DeepGlobe'
    if folder.endswith('tuning'):
        key += '_tuning'
    elif folder.endswith('evaluation'):
        key += '_segtest'
    if 'vgg16' in folder:
        key += '_VGG16'
    elif 'x1.7' in folder:
        key += '_X1.7'
    elif 'm7' in folder:
        key += '_M7'
    return key
for folder in folders:
    key = irn_folder_to_key(folder)
    if 'cam' in folder:
        fname = folder + '_cam_iou.csv'
        df = pd.read_csv(os.path.join(DIR, folder, fname))
        eval_cam[key] = df[df['Unnamed: 0'] == 'mean']['iou'].values[0]
    else:
        fname = folder + '_iou.csv'
        df = pd.read_csv(os.path.join(DIR, folder, fname))
        eval_irn[key] = df[df['Unnamed: 0'] == 'miou']['iou'].values[0]

# HistoSegNet
DIR = '../03c_hsn/eval'
folders = os.listdir(DIR)
eval_hsn = {}

for folder in folders:
    assert folder in SETS
    fnames = [x for x in os.listdir(os.path.join(DIR, folder)) if x.endswith('.xlsx') and not x.startswith('~')]
    assert len(fnames) == 1
    fname = fnames[0]
    df = pd.read_excel(os.path.join(DIR, folder, fname))
    eval_hsn[folder] = df['IoU'][df['Class'] == 'Mean'].values[0]

df_eval = pd.DataFrame({'Grad-CAM': eval_cam, 'SEC': eval_sec_dsrg['SEC'], 'DSRG': eval_sec_dsrg['DSRG'],
                        'IRNet': eval_irn, 'HistoSegNet': eval_hsn})
pd.set_option('display.max_columns', None)
print(df_eval)
a=1