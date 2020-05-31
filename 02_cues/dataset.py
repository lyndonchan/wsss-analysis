import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

class Dataset:
    """Class for implementing dataset handling"""

    def __init__(self, data_type='ADP', size=321, batch_size=16, first_inds=None):
        self.data_type = data_type
        self.size = size
        self.batch_size = batch_size
        self.database_dir = os.path.join(os.path.dirname(os.getcwd()), 'database')
        #
        self.load_attributes()
        self.load_data(first_inds)

    def load_attributes(self):
        """Load dataset attributes, especially ImageDataGenerator"""

        if self.data_type == 'ADP':
            self.devkit_dir = os.path.join(self.database_dir, 'ADPdevkit', 'ADPRelease1')
            self.sets = ['valid', 'test']
            self.is_evals = [True, True]
            self.class_names = ['E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I', 'C.D.R', 'C.L', 'H.E',
                           'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W', 'A.B', 'A.M', 'M.M', 'M.K',
                           'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W', 'G.O', 'G.N', 'T']  # 31 classes

            def normalize(x):
                x = (x - 193.09203) / (56.450138 + 1e-7)
                return x

            self.datagen_aug = ImageDataGenerator(
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
            self.datagen_nonaug = ImageDataGenerator(
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
        elif self.data_type == 'VOC2012':
            self.devkit_dir = os.path.join(self.database_dir, 'VOCdevkit', 'VOC_trainaug_val', 'VOC2012')
            self.sets = ['trainaug', 'val']
            self.is_evals = [False, True]
            self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train',
                           'tvmonitor']  # 20 classes

            def normalize(x):
                x[:, :, 0] -= 104
                x[:, :, 1] -= 117
                x[:, :, 2] -= 123
                return x

            self.datagen_aug = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.2,
                rotation_range=30,
                fill_mode='reflect',
                preprocessing_function=normalize)
            self.datagen_nonaug = ImageDataGenerator(rescale=1. / 255,
                                                preprocessing_function=normalize)
        elif 'DeepGlobe' in self.data_type:
            self.devkit_dir = os.path.join(self.database_dir, 'DGdevkit')
            if self.data_type == 'DeepGlobe':
                self.sets = ['train75', 'test']
                self.is_evals = [False, True]
            elif self.data_type == 'DeepGlobe_balanced':
                self.sets = ['train37.5', 'test']
                self.is_evals = [False, True]
            self.class_names = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren', 'unknown']  # 7 classes
            self.datagen_aug = ImageDataGenerator(
                rescale=1. / 255,
                vertical_flip=True,
                horizontal_flip=True)
            self.datagen_nonaug = ImageDataGenerator(
                rescale=1. / 255)

    def load_data(self, first_inds):
        """Load DataFrameIterator for dataset"""
        assert type(first_inds) == int
        self.set_gens = {}
        if self.data_type == 'ADP':
            img_folder = 'PNGImages'
        else:
            img_folder = 'JPEGImages'

        for s, is_eval in zip(self.sets, self.is_evals):
            set_df = pd.read_csv(os.path.join(self.devkit_dir, 'ImageSets', 'Segmentation', s + '.csv'))
            if first_inds is not None:
                set_df = set_df.iloc[:min(first_inds, set_df.shape[0]), :] #
            if not is_eval:
                self.set_gens[s] = self.datagen_aug.flow_from_dataframe(dataframe=set_df,
                                                              directory=os.path.join(self.devkit_dir, img_folder),
                                                              x_col='Patch Names',
                                                              y_col=self.class_names,
                                                              batch_size=self.batch_size,
                                                              class_mode='other',
                                                              target_size=(self.size, self.size),
                                                              shuffle=True)
            else:
                self.set_gens[s] = self.datagen_nonaug.flow_from_dataframe(dataframe=set_df,
                                                                 directory=os.path.join(self.devkit_dir, img_folder),
                                                                 x_col='Patch Names',
                                                                 y_col=self.class_names,
                                                                 batch_size=self.batch_size,
                                                                 class_mode='other',
                                                                 target_size=(self.size, self.size),
                                                                 shuffle=False)