import os

DIR = r'C:\Users\chanlynd\Documents\Grad Research\wsss-analysis\database\models_cnn_full'
folders = os.listdir(DIR)
for folder in folders:
    for file in os.listdir(os.path.join(DIR, folder)):
        if '_wpt' not in file:
            os.rename(os.path.join(DIR, folder, file), os.path.join(DIR, folder, os.path.splitext(file)[0] + '_wpt' +
                                                                    os.path.splitext(file)[1]))
    if '_wpt' not in folder:
        os.rename(os.path.join(DIR, folder), os.path.join(DIR, folder + '_wpt'))