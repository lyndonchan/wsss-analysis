from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os

def load_data_VOC2012(db_dir, size, batch_size):
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']  # 20 classes
    # Training set dataframe
    trainaug_df = pd.read_csv(os.path.join(db_dir, 'VOCdevkit', 'VOC_trainaug_val', 'VOC2012', 'ImageSets', 'Segmentation', 'trainaug.csv'))
    # Validation set dataframe
    valid_df = pd.read_csv(os.path.join(db_dir, 'VOCdevkit', 'VOC_trainaug_val', 'VOC2012', 'ImageSets', 'Segmentation', 'val.csv'))
    # Test set dataframe
    with open(os.path.join(db_dir, 'VOCdevkit', 'VOC_test', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')) as f:
        test_lines = []
        for line in f.readlines():
            test_lines.append(line.strip('\n').split('.jpg')[0] + '.jpg')
        test_df = pd.DataFrame(test_lines)
    test_df.columns = ['filename']

    # Get train class counts
    train_class_counts = [x for i, x in enumerate(np.sum(trainaug_df.values[:, 1:], axis=0)) if
                          trainaug_df.columns[i + 1] in class_names]

    # Set up data generators
    def normalize(x):
        x[:, :, 0] -= 104
        x[:, :, 1] -= 117
        x[:, :, 2] -= 123
        return x
    datagen_aug = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        rotation_range=30,
        fill_mode='reflect',
        preprocessing_function=normalize)
    datagen_nonaug = ImageDataGenerator(rescale=1./255,
                                        preprocessing_function=normalize)

    devkit_dir = os.path.join(db_dir, 'VOCdevkit', 'VOC_trainaug_val', 'VOC2012')
    test_dir = os.path.join(db_dir, 'VOCdevkit', 'VOC_test', 'VOC2012')
    train_generator = datagen_aug.flow_from_dataframe(dataframe=trainaug_df,
                                                      directory=os.path.join(devkit_dir, 'JPEGImages'),
                                                      x_col='Patch Names',
                                                      y_col=class_names,
                                                      batch_size=batch_size,
                                                      class_mode='other',
                                                      target_size=(size[0], size[1]),
                                                      shuffle=True)
    valid_generator = datagen_nonaug.flow_from_dataframe(dataframe=valid_df,
                                                         directory=os.path.join(devkit_dir, 'JPEGImages'),
                                                         x_col='Patch Names',
                                                         y_col=class_names,
                                                         batch_size=batch_size,
                                                         class_mode='other',
                                                         target_size=(size[0], size[1]),
                                                         shuffle=False)
    test_generator = datagen_nonaug.flow_from_dataframe(dataframe=test_df,
                                                        directory=os.path.join(test_dir, 'JPEGImages'),
                                                        x_col='filename',
                                                        batch_size=batch_size,
                                                        class_mode='input',
                                                        target_size=(size[0], size[1]),
                                                        shuffle=False)

    return train_generator, valid_generator, test_generator, class_names, train_class_counts

def load_data_ADP(db_dir, size, batch_size, is_aug):
    if not is_aug:
        class_names = ['E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I', 'C.D.R', 'C.L', 'H.E',
                   'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W', 'A.B', 'A.M', 'M.M', 'M.K',
                   'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W', 'G.O', 'G.N', 'T']  # 31 classes
    else:
        class_names = ['E', 'E.M', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C', 'C.D',
                       'C.D.I', 'C.D.R', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.M.C', 'S.M.S', 'S.E', 'S.C',
                       'S.C.H', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K', 'N', 'N.P', 'N.R', 'N.R.B',
                       'N.R.A', 'N.G', 'N.G.M', 'N.G.A', 'N.G.O', 'N.G.E', 'N.G.R', 'N.G.W', 'N.G.T', 'G', 'G.O',
                       'G.N', 'T']

    csv_path = os.path.join(db_dir, 'ADPdevkit', 'ADPRelease1', 'labels', 'ADP_EncodedLabels_Release1.csv')
    img_dir = os.path.join(db_dir, 'ADPdevkit', 'ADPRelease1', 'PNGImages')
    splits_dir = os.path.join(db_dir, 'ADPdevkit', 'ADPRelease1', 'splits')
    all_df = pd.read_csv(csv_path)
    # Read splits
    train_inds = np.load(os.path.join(splits_dir, 'train.npy'))
    valid_inds = np.load(os.path.join(splits_dir, 'valid.npy'))
    test_inds = np.load(os.path.join(splits_dir, 'test.npy'))
    # Split dataframe
    train_df = all_df.loc[train_inds, :]
    valid_df = all_df.loc[valid_inds, :]
    test_df = all_df.loc[test_inds, :]
    # Get train class counts
    train_class_counts = [x for i, x in enumerate(np.sum(train_df.values[:, 1:], axis=0)) if
                          train_df.columns[i + 1] in class_names]

    # Set up data generators
    def normalize(x):
        x = (x - 193.09203) / (56.450138 + 1e-7)
        return x
    datagen_aug = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        preprocessing_function=normalize)  # normalize by subtracting training set image mean, dividing by training set image std
    datagen_nonaug = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=normalize)  # normalize by subtracting training set image mean, dividing by training set image std
    train_generator = datagen_aug.flow_from_dataframe(dataframe=train_df,
                                                      directory=img_dir,
                                                      x_col='Patch Names',
                                                      y_col=class_names,
                                                      batch_size=batch_size,
                                                      class_mode='other',
                                                      target_size=(size[0], size[1]),
                                                      shuffle=True)
    valid_generator = datagen_nonaug.flow_from_dataframe(dataframe=valid_df,
                                                         directory=img_dir,
                                                         x_col='Patch Names',
                                                         y_col=class_names,
                                                         batch_size=batch_size,
                                                         class_mode='other',
                                                         target_size=(size[0], size[1]),
                                                         shuffle=False)
    test_generator = datagen_nonaug.flow_from_dataframe(dataframe=test_df,
                                                        directory=img_dir,
                                                        x_col='Patch Names',
                                                        y_col=class_names,
                                                        batch_size=batch_size,
                                                        class_mode='other',
                                                        target_size=(size[0], size[1]),
                                                        shuffle=False)
    return train_generator, valid_generator, test_generator, class_names, train_class_counts

def load_data_DeepGlobe(db_dir, size, batch_size, is_balanced):
    class_names = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren', 'unknown'] # 7 classes

    # Training set dataframe
    if not is_balanced:
        train_df = pd.read_csv(os.path.join(db_dir, 'DGdevkit', 'ImageSets', 'Segmentation', 'train75.csv'))
    else:
        train_df = pd.read_csv(os.path.join(db_dir, 'DGdevkit', 'ImageSets', 'Segmentation', 'train37.5.csv'))
    test_df = pd.read_csv(os.path.join(db_dir, 'DGdevkit', 'ImageSets', 'Segmentation', 'test.csv'))

    # Get train class counts
    train_class_counts = [x for i, x in enumerate(np.sum(train_df.values[:, 1:], axis=0)) if
                          train_df.columns[i + 1] in class_names]

    # Set up data generators
    datagen_aug = ImageDataGenerator(
        rescale=1./255,
        vertical_flip=True,
        horizontal_flip=True)
    datagen_nonaug = ImageDataGenerator(
        rescale=1. / 255)

    train_generator = datagen_aug.flow_from_dataframe(dataframe=train_df,
                                                      directory=os.path.join(db_dir, 'DGdevkit', 'JPEGImages'),
                                                      x_col='Patch Names',
                                                      y_col=class_names,
                                                      batch_size=batch_size,
                                                      class_mode='other',
                                                      target_size=(size[0], size[1]),
                                                      shuffle=True)
    valid_generator = datagen_nonaug.flow_from_dataframe(dataframe=test_df,
                                                        directory=os.path.join(db_dir, 'DGdevkit', 'JPEGImages'),
                                                        x_col='Patch Names',
                                                        y_col=class_names,
                                                        batch_size=batch_size,
                                                        class_mode='other',
                                                        target_size=(size[0], size[1]),
                                                        shuffle=False)
    test_generator = datagen_nonaug.flow_from_dataframe(dataframe=test_df,
                                                        directory=os.path.join(db_dir, 'DGdevkit', 'JPEGImages'),
                                                        x_col='Patch Names',
                                                        y_col=class_names,
                                                        batch_size=batch_size,
                                                        class_mode='other',
                                                        target_size=(size[0], size[1]),
                                                        shuffle=False)

    return train_generator, valid_generator, test_generator, class_names, train_class_counts