import os

DIR = '../database/models_wsss/DSRG'
folders = [x for x in os.listdir(DIR) if not os.path.isfile(os.path.join(DIR, x))]
for folder in folders:
    for file in os.listdir(os.path.join(DIR, folder)):
        if '_train75_' in file:
            os.rename(os.path.join(DIR, folder, file), os.path.join(DIR, folder, file.replace('_train75_', '_')))
        elif '_train37.5_' in file:
            os.rename(os.path.join(DIR, folder, file), os.path.join(DIR, folder, file.replace('_train37.5_', '_balanced_')))
    if '_train75_' in folder:
        os.rename(os.path.join(DIR, folder), os.path.join(DIR, folder.replace('_train75_', '_')))
    elif '_train37.5_' in folder:
        os.rename(os.path.join(DIR, folder), os.path.join(DIR, folder.replace('_train37.5_', '_balanced_')))