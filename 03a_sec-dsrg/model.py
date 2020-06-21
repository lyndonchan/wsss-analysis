import os
import time
import pickle
import cv2
import configparser
import tensorflow as tf
import skimage.color as imgco
import skimage.io as imgio
import multiprocessing
import pandas as pd
import numpy.matlib
import traceback

from utilities import *
from lib.crf import crf_inference

from DSRG import DSRG
from SEC import SEC

config = configparser.ConfigParser()
config.read('../settings.ini')
DATA_ROOT = config['Download Directory']['data_dir']
WSSS_MODEL_ROOT = os.path.join(config['Download Directory']['data_dir'], config['Data Folders']['model_wsss_dir'])
CUES_DIR = os.path.join(config['Download Directory']['data_dir'], config['Data Folders']['cues_dir'])

class Model():
    """Wrapper class for SEC and DSRG WSSS methods"""

    def __init__(self, args):
        self.task = args.task
        self.method = args.method
        self.dataset = args.dataset
        self.seed_type = args.seed
        self.h, self.w = (321, 321)
        self.seed_size = 41
        self.weight_decay = 5e-4
        self.base_lr = 1e-4
        self.lr_decay = 0.5
        self.lr_step = 4
        self.momentum = 0.9
        if self.task == 'train':
            self.max_epochs = args.epochs
        self.threshold = args.threshold
        self.batch_size = args.batchsize
        self.should_saveimg = args.saveimg
        self.verbose = args.verbose

        if self.dataset in ['ADP-morph', 'ADP-func']:
            self.eval_setname = args.eval_setname
        if self.task == 'predict':
            if self.dataset in ['ADP-morph', 'ADP-func']:
                self.phase = self.eval_setname
            elif self.dataset == 'VOC2012':
                self.phase = 'val'
            elif 'DeepGlobe' in self.dataset:
                self.phase = 'test'
        elif self.task == 'train':
            self.phase = 'train'
        # paths
        self.save_dir = os.path.join(WSSS_MODEL_ROOT, self.method, self.dataset + '_' + self.seed_type)
        self.out_dir = os.path.join('out', self.method, self.dataset + '_train_' + self.seed_type)
        self.eval_dir = os.path.join('eval', self.method, self.dataset + '_train_' + self.seed_type)
        new_dirs = [self.save_dir, self.eval_dir]
        if self.task == 'train':
            self.log_dir = os.path.join('log', self.method, self.dataset + '_train_' + self.seed_type)
            new_dirs.append(self.log_dir)
        for pth in new_dirs:
            if not os.path.exists(pth):
                os.makedirs(pth)

        self.accum_num = 1
        self.pool = multiprocessing.Pool()

        self.saver = {}

        self.l2loss = {"total": 0}

        if self.method == 'DSRG':
            self.init_model_path = os.path.join(WSSS_MODEL_ROOT, self.method, 'vgg16_deeplab_aspp.npy')
        elif self.method == 'SEC':
            self.init_model_path = os.path.join(WSSS_MODEL_ROOT, self.method, 'init.npy')

        if self.dataset == 'ADP-morph':
            self.num_classes = 29
            self.img_mean = np.array([208.8502, 163.2828, 207.1458])
            self.dataset_dir = os.path.join(DATA_ROOT, 'ADPdevkit', 'ADPRelease1')
            self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list.txt')
            if self.task == 'train':
                self.run_categories = ['tuning', 'segtest']
            elif self.task == 'predict':
                self.run_categories = [self.eval_setname]
            self.class_names = ['Background', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I',
                                'C.D.R', 'C.L', 'H.E', 'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W',
                                'A.B', 'A.M', 'M.M', 'M.K', 'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W']
            self.label2rgb_colors = np.array([(255, 255, 255), (0, 0, 128), (0, 128, 0), (255, 165, 0), (255, 192, 203),
                                              (255, 0, 0), (173, 20, 87), (176, 141, 105), (3, 155, 229),
                                              (158, 105, 175), (216, 27, 96), (244, 81, 30), (124, 179, 66),
                                              (142, 36, 255), (240, 147, 0), (204, 25, 165), (121, 85, 72),
                                              (142, 36, 170), (179, 157, 219), (121, 134, 203), (97, 97, 97),
                                              (167, 155, 142), (228, 196, 136), (213, 0, 0), (4, 58, 236),
                                              (0, 150, 136), (228, 196, 65), (239, 108, 0), (74, 21, 209)])
        elif self.dataset == 'ADP-func':
            self.num_classes = 5
            self.img_mean = np.array([208.8502, 163.2828, 207.1458])
            self.dataset_dir = os.path.join(DATA_ROOT, 'ADPdevkit', 'ADPRelease1')
            self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list.txt')
            if self.task == 'train':
                self.run_categories = ['tuning', 'segtest']
            elif self.task == 'predict':
                self.run_categories = [self.eval_setname]
            self.class_names = ['Background', 'Other', 'G.O', 'G.N', 'T']
            self.label2rgb_colors = np.array([(255, 255, 255), (3, 155, 229), (0, 0, 128), (0, 128, 0), (173, 20, 87)])
        elif self.dataset == 'VOC2012':
            self.num_classes = 21
            self.img_mean = np.array([104.00698793, 116.66876762, 122.67891434])
            self.dataset_dir = os.path.join(DATA_ROOT, 'VOCdevkit', 'VOC_trainaug_val', 'VOC2012')
            self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list.txt')
            self.run_categories = ['val']
            self.class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor']
            self.label2rgb_colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                                              (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                                              (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                                              (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                              (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                                              (0, 64, 128)])
        elif 'DeepGlobe' in self.dataset:
            self.num_classes = 6
            self.img_mean = np.array([0.0, 0.0, 0.0])
            self.dataset_dir = os.path.join(DATA_ROOT, 'DGdevkit')
            if self.dataset == 'DeepGlobe':
                self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list_train75.txt')
            elif self.dataset == 'DeepGlobe_balanced':
                self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list_train37.5.txt')
            self.run_categories = ['test']
            self.class_names = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren']
            self.label2rgb_colors = np.array([(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255),
                        (255, 255, 255)])
        self.img_mean = np.expand_dims(np.expand_dims(self.img_mean, axis=0), axis=1)

        # For dataset
        self.ignore_label = 255

        self.run_categories = ['train'] + self.run_categories
        self.data_f, self.data_len = self.get_data_f(self.run_categories)
    def load(self):
        """Load either SEC or DSRG model"""
        if self.method == 'DSRG':
            self.model = DSRG({'dataset': self.dataset, 'img_size': self.h, 'num_classes': self.num_classes,
                               'batch_size': self.batch_size, 'phase': self.phase, 'img_mean': self.img_mean,
                               'seed_size': self.seed_size, 'pool': self.pool, 'init_model_path': self.init_model_path})
        elif self.method == 'SEC':
            self.model = SEC({'dataset': self.dataset, 'img_size': self.h, 'num_classes': self.num_classes,
                              'batch_size': self.batch_size, 'phase': self.phase, 'img_mean': self.img_mean,
                              'seed_size': self.seed_size, 'pool': self.pool, 'init_model_path': self.init_model_path})
    def get_data_f(self, phases):
        """Load names of files in the dataset

        Parameters
        ----------
        phases : list of str
            The phases to run the Model object on
        """
        # cues_labels = [self.cues_data[x] for i, x in enumerate(self.cues_data.keys()) if '_labels' in x]
        # [i for i,x in enumerate(cues_labels) if 32 in x]
        data_f = {}
        data_len = {}
        for phase in phases:
            data_f[phase] = {"img": [], "label": [], "id": [], "id_for_slice": []}
            data_len[phase] = 0
            if phase in ['train']:
                if 'ADP' in self.dataset:
                    cues_path = os.path.join(CUES_DIR, 'ADP_' + self.seed_type, self.dataset.split('-')[-1],
                                             'localization_cues.pickle')
                else:
                    cues_path = os.path.join(CUES_DIR, self.dataset + '_' + self.seed_type, 'localization_cues.pickle')
                self.cues_data = pickle.load(open(cues_path, "rb"), encoding="iso-8859-1")
                with open(self.input_path, "r") as f:
                    for line in f.readlines():
                        line = line.rstrip("\n")
                        id_name,id_identy = line.split(" ")
                        id_name = id_name[:-4] # then id_name is like '2007_007028'
                        data_f[phase]["id"].append(id_name)
                        data_f[phase]["id_for_slice"].append(id_identy)
                        data_f[phase]["img"].append(os.path.join(self.dataset_dir,"JPEGImages","%s.jpg" % id_name))
                data_len[phase] = len(data_f[phase]["id"])
            elif phase in ['val', 'tuning', 'segtest', 'test']:
                val_file_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', "%s.txt" % phase)
                with open(val_file_path, "r") as f:
                    for line in f.readlines():
                        line = line.strip("\n")  # the line is like "2007_000738"
                        data_f[phase]["id"].append(line)
                        data_f[phase]["img"].append(os.path.join(self.dataset_dir, "JPEGImages", "%s.jpg" % line))
                        if self.dataset == 'VOC2012' or 'DeepGlobe' in self.dataset:
                            data_f[phase]["label"].append(
                               os.path.join(self.dataset_dir, "SegmentationClassAug", "%s.png" % line))
                        elif self.dataset in ['ADP-morph', 'ADP-func']:
                            data_f[phase]["label"].append(
                                os.path.join(self.dataset_dir, "SegmentationClassAug", self.dataset, "%s.png" % line))
                data_len[phase] = len(data_f[phase]["id"])
                data_len[phase] = len(data_f[phase]["label"])
        if self.verbose:
            print("len:%s" % str(data_len))
        return data_f,data_len

    def next_batch(self,category=None,max_epochs=-1):
        """Load next batch in the dataset for running

        Parameters
        ----------
        category : str, optional
            The category to run the Model object in
        max_epochs : int, optional
            The maximum number of epochs to run the Model object before terminating

        Returns
        -------
        img : Tensor
            Input images in batch, after resizing and normalizing
        label : Tensor
            GT segmentation in batch, after resizing
        id : Tensor
            Filenames in batch
        iterator : Iterator
            Iterator object through dataset
        """
        self.category = category
        def m_train(x):
            id_ = x["id"]
            img_f = x["img_f"]
            img_raw = tf.read_file(img_f)
            img = tf.image.decode_image(img_raw)
            img, _ = self.image_preprocess(img)
            img = tf.reshape(img, [self.h, self.w, 3])
            id_for_slice = x["id_for_slice"]

            def get_data(identy):
                identy = identy.decode()
                label = np.zeros([self.num_classes])
                label[self.cues_data["%s_labels" % identy]] = 1.0
                label[0] = 1.0
                cues = np.zeros([self.seed_size, self.seed_size, self.num_classes])
                cues_i = self.cues_data["%s_cues" % identy]
                cues[cues_i[1], cues_i[2], cues_i[0]] = 1.0
                return label.astype(np.float32), cues.astype(np.float32)

            label, cues = tf.py_func(get_data, [id_for_slice], [tf.float32, tf.float32])
            label.set_shape([self.num_classes])
            cues.set_shape([self.seed_size, self.seed_size, self.num_classes])

            return img, label, cues, id_

        def m_val(x):
            id = x["id"]
            img_f = x["img_f"]
            img_raw = tf.read_file(img_f)
            img = tf.image.decode_image(img_raw)
            img = tf.expand_dims(img, axis=0)
            label_f = x["label_f"]
            label_raw = tf.read_file(label_f)
            label = tf.image.decode_image(label_raw)
            label = tf.expand_dims(label, axis=0)
            img, label = self.image_preprocess(img, label)
            if self.h is not None:
                img = tf.reshape(img, [self.h, self.w, 3])
                label = tf.reshape(label, [self.h, self.w, 3])
            label = tf.cast(label, tf.int32)

            return img, label, id

        if category in ['train', 'debug']:
            dataset = tf.data.Dataset.from_tensor_slices({
                "id":self.data_f[category]["id"],
                "id_for_slice":self.data_f[category]["id_for_slice"],
                "img_f":self.data_f[category]["img"],
                })
            dataset = dataset.repeat(max_epochs)
            dataset = dataset.shuffle(self.data_len[category])
            dataset = dataset.map(m_train)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            img, label, cues, id_ = iterator.get_next()

            return img, label, cues, id_, iterator

        elif category in ['val', 'tuning', 'segtest', 'test']:
            dataset = tf.data.Dataset.from_tensor_slices({
                "id": self.data_f[category]["id"],
                "img_f": self.data_f[category]["img"],
                "label_f": self.data_f[category]["label"]
            })
            dataset = dataset.repeat(max_epochs)
            dataset = dataset.map(m_val)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            img, label, id = iterator.get_next()

            return img, label, id, iterator

    def image_preprocess(self, img, label=None):
        """Load next batch in the dataset for running

        Parameters
        ----------
        img : Tensor
            Input images in batch, before resizing and normalizing
        label : Tensor, optional
            GT segmentation in batch, before resizing

        Returns
        -------
        img : Tensor
            Input images in batch, after resizing and normalizing
        label : Tensor
            GT segmentation in batch, after resizing
        """
        if self.category in ['train', 'debug']:
            img = tf.expand_dims(img,axis=0)
            img = tf.image.resize_bilinear(img,(self.h,self.w))
            img = tf.squeeze(img,axis=0)
            if label is not None:
                label = tf.expand_dims(label,axis=0)
                label = tf.image.resize_nearest_neighbor(label,(self.h,self.w))
                label = tf.squeeze(label,axis=0)

            r,g,b = tf.split(axis=2,num_or_size_splits=3,value=img)
            img = tf.cast(tf.concat([b,g,r],2),dtype=tf.float32)
            img -= self.img_mean

            return img,label
        elif self.category in ['val', 'tuning', 'segtest', 'test']:
            img = tf.squeeze(img, squeeze_dims=[0])
            img = tf.expand_dims(img, axis=0)
            img = tf.image.resize_bilinear(img, (self.h, self.w))
            img = tf.squeeze(img, axis=0)

            if label is not None:
                label = tf.squeeze(label, squeeze_dims=[0])
                label = tf.expand_dims(label, axis=0)
                label = tf.image.resize_nearest_neighbor(label, (self.h, self.w))
                label = tf.squeeze(label, axis=0)

            r, g, b = tf.split(axis=2, num_or_size_splits=3, value=img)
            img = tf.cast(tf.concat([b, g, r], 2), dtype=tf.float32)
            img -= self.img_mean

            return img, label

    def label2rgb(self, label, category_num, colors=[], ignore_label=128, ignore_color=(255,255,255)):
        """Convert index labels to RGB colour images

        Parameters
        ----------
        label : numpy 2D array (size: H x W)
            The index label map, with each array element equal to the class index at that position
        category_num : int
            The number of classes
        colours : numpy 1D array of 3-tuples of int
            List of colours for each class
        ignore_label : int
            Class index to ignore as background
        ignore_color : 3-tuple of int
            Class colour to ignore as background

        Returns
        -------
        (label_uint8) : numpy 2D array (size: H x W x 3)
            The label map, as a colour RGB image
        """

        if len(colors) <= 0:
            index = np.unique(label)
            index = index[index < category_num]
            colors = self.label2rgb_colors[index]
        label = imgco.label2rgb(label, colors=colors, bg_label=ignore_label, bg_color=ignore_color)
        return label.astype(np.uint8)

    def optimize(self):
        for val_category in self.run_categories[1:]:
            self.model.metric["miou_" + val_category] = tf.Variable(0.0, trainable=False)
        self.model.loss["norm"] = self.model.getloss()
        self.model.loss["l2"] = sum([tf.nn.l2_loss(self.model.weights[layer][0]) for layer in self.model.weights])
        self.model.loss["total"] = self.model.loss["norm"]+ self.weight_decay * self.model.loss["l2"]
        self.model.net["lr"] = tf.Variable(self.base_lr, trainable=False)
        opt = tf.train.MomentumOptimizer(self.model.net["lr"], self.momentum)
        gradients = opt.compute_gradients(self.model.loss["total"])
        self.model.net["accum_gradient"] = []
        self.model.net["accum_gradient_accum"] = []
        new_gradients = []
        for (g,v) in gradients:
            if g is None: continue
            if v in self.model.lr_2_list:
                g = 2*g
            if v in self.model.lr_10_list:
                g = 10*g
            if v in self.model.lr_20_list:
                g = 20*g
            self.model.net["accum_gradient"].append(tf.Variable(tf.zeros_like(g),trainable=False))
            self.model.net["accum_gradient_accum"].append(self.model.net["accum_gradient"][-1].assign_add( g/self.accum_num, use_locking=True))
            new_gradients.append((self.model.net["accum_gradient"][-1],v))

        self.model.net["accum_gradient_clean"] = [g.assign(tf.zeros_like(g)) for g in self.model.net["accum_gradient"]]
        self.model.net["accum_gradient_update"]  = opt.apply_gradients(new_gradients)

    def get_latest_checkpoint(self):
        """Find the filepath to the saved model checkpoint"""
        if 'final-0.index' in os.listdir(self.save_dir):
            return os.path.join(self.save_dir, 'final-0')
        all_checkpoint_inds = [int(x.split('.')[0].split('-')[-1]) for x in os.listdir(self.save_dir) if
                                 'epoch-' in x and '.index' in x]
        if len(all_checkpoint_inds) == 0:
            return None
        latest_checkpoint = os.path.join(self.save_dir, 'epoch-' + str(max(all_checkpoint_inds)))
        return latest_checkpoint

    def restore_from_model(self, saver, model_path, checkpoint=False):
        """Restore model from saved checkpoint

        Parameters
        ----------
        saver : Saver
            The Saver object used to load the model
        model_path : str
            The file path to the saved checkpoint
        """

        assert self.sess is not None
        if checkpoint is True:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(self.sess, model_path)

    def train(self):
        tf.reset_default_graph()
        # self.sess = tf.Session()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        data_x,data_label,data_cues,id_of_image,iterator_train = self.next_batch(category="train", max_epochs=-1)
        data_x_val = {}
        data_label_val = {}
        id_of_image_val = {}
        iterator_val = {}
        for val_category in self.run_categories[1:]:
            data_x_val[val_category], data_label_val[val_category], id_of_image_val[val_category], \
            iterator_val[val_category] = self.next_batch(category=val_category, max_epochs=1)
        self.model.build(net_input=data_x, net_label=data_label, net_cues=data_cues, net_id=id_of_image, phase='train')
        self.optimize()
        self.saver["epoch"] = tf.train.Saver(max_to_keep=self.max_epochs,var_list=self.model.trainable_list)
        self.saver["final"] = tf.train.Saver(max_to_keep=1,var_list=self.model.trainable_list)

        iterations_per_epoch_train = self.data_len['train'] // self.batch_size  # self.data.get_data_len()

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)
            for val_category in self.run_categories[1:]:
                self.sess.run(iterator_val[val_category].initializer)
            tf.summary.scalar("seed_loss", self.model.loss["seed"])
            tf.summary.scalar("constrain_loss", self.model.loss["constrain"])
            tf.summary.scalar("total_loss", self.model.loss["total"])
            tf.summary.scalar("loss", self.model.loss["norm"])
            for val_category in self.run_categories[1:]:
                tf.summary.scalar("miou_" + val_category, self.model.metric["miou_" + val_category])
            tf.summary.scalar("lr", self.model.net["lr"])
            tf.summary.scalar("epoch", self.model.net["epoch"])
            merged_summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

            # Resume training from latest checkpoint if it exists
            latest_ckpt = self.get_latest_checkpoint()
            if latest_ckpt is not None:
                print('Loading model from previous checkpoint %s' % latest_ckpt)
                print("before l2 loss:%f" % self.sess.run(self.model.loss["l2"]))
                self.restore_from_model(self.saver["epoch"], latest_ckpt, checkpoint=False)
                print("after l2 loss:%f" % self.sess.run(self.model.loss["l2"]))
                i = int(os.path.basename(latest_ckpt).split('-')[-1])
                epoch = i / iterations_per_epoch_train
            # Start training from scratch if no previous checkpoint
            else:
                epoch, i = 0.0, 0
                self.sess.run(self.model.net["accum_gradient_clean"])
            seed_losses, constrain_losses, total_losses = [], [], []

            while epoch < self.max_epochs:
                start_time = time.time()
                lr = self.base_lr * self.lr_decay ** (epoch // self.lr_step)
                if i % iterations_per_epoch_train == 0:
                    self.sess.run(tf.assign(self.model.net["lr"], lr))

                self.sess.run(self.model.net["accum_gradient_accum"])
                seed_l, constrain_l, loss, lr = self.sess.run(
                    [self.model.loss["seed"], self.model.loss["constrain"], self.model.loss["total"], self.model.net["lr"]])
                seed_losses.append(seed_l), constrain_losses.append(constrain_l), total_losses.append(loss)

                if i % self.accum_num == self.accum_num - 1:
                    self.sess.run(self.model.net["accum_gradient_update"])
                    self.sess.run(self.model.net["accum_gradient_clean"])
                if i % 200 == 0:
                    # Get small training losses
                    dbg_s = np.mean(np.array(seed_losses))
                    dbg_c = np.mean(np.array(constrain_losses))
                    dbg_l = np.mean(np.array(total_losses))
                    seed_losses, constrain_losses, total_losses = [], [], []
                    print('epoch=%f | seed_loss=%f, constrain_loss=%f, total_loss=%f' % (epoch, dbg_s, dbg_c, dbg_l))
                    # Create save image directory if non-existent
                    if self.should_saveimg:
                        saveimg_dir = os.path.join(self.out_dir, 'train', str(i))
                        if not os.path.exists(saveimg_dir):
                            os.makedirs(saveimg_dir)
                    else:
                        saveimg_dir = None
                    # Get validation metric
                    for val_category in self.run_categories[1:]:
                        miou_val = self.eval_miou(data_x_val[val_category], id_of_image_val[val_category],
                                                  data_label_val[val_category], self.model.net['input'],
                                                  self.model.net['rescale_output'], self.model.net['drop_prob'],
                                                  val_category, saveimg_dir)
                        self.sess.run(tf.assign(self.model.metric["miou_" + val_category], miou_val))
                        self.sess.run(iterator_val[val_category].initializer)
                        print('mIoU [%s] = %f' % (val_category, miou_val))
                    self.sess.run(tf.assign(self.model.net["epoch"], epoch))
                    # Update Tensorboard
                    summary = self.sess.run(merged_summary_op)
                    summary_writer.add_summary(summary, i)
                i+=1
                epoch = i / iterations_per_epoch_train
                print('Image run-time: %f' % ((time.time() - start_time) / self.batch_size))

                if i % iterations_per_epoch_train == 0:
                    self.saver["epoch"].save(self.sess, os.path.join(self.save_dir, "epoch"), global_step=i)
            self.saver["final"].save(self.sess,os.path.join(self.save_dir, "final"), global_step=0)
            self.pool.close()
            self.pool.join()

    def predict(self):
        """Predict the segmentation for the requested dataset"""
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        data_x = {}
        data_label = {}
        id_of_image = {}
        iterator = {}
        for val_category in self.run_categories[1:]:
            data_x[val_category], data_label[val_category], id_of_image[val_category], \
            iterator[val_category] = self.next_batch(category=val_category, max_epochs=1)
        first_cat = self.run_categories[1]
        self.model.build(net_input=data_x[first_cat], net_label=data_label[first_cat], net_id=id_of_image[first_cat],
                         phase=first_cat)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        for val_category in self.run_categories[1:]:
            self.sess.run(iterator[val_category].initializer)

        # Resume training from latest checkpoint if it exists
        self.saver = tf.train.Saver(max_to_keep=1, var_list=self.model.trainable_list)
        latest_ckpt = self.get_latest_checkpoint()
        if latest_ckpt is not None:
            if self.verbose:
                print('Loading model from previous checkpoint %s' % latest_ckpt)
            self.restore_from_model(self.saver, latest_ckpt)

        # Create save image directory if non-existent
        layers = ['rescale_output']
        for layer in layers:
            for val_category in self.run_categories[1:]:
                saveimg_dir = os.path.join(self.out_dir, 'predict_' + val_category, layer)
                if not os.path.exists(saveimg_dir):
                    os.makedirs(saveimg_dir)
                miou_val = self.eval_miou(data_x[val_category], id_of_image[val_category], data_label[val_category],
                                          self.model.net['input'], self.model.net[layer], self.model.net['drop_prob'],
                                          val_category, saveimg_dir, should_overlay=True, is_eval=True)
                if self.verbose:
                    print('mIoU [%s] = %f' % (val_category, miou_val))
        self.pool.close()
        self.pool.join()

    def single_crf_metrics(self, params):
        """Run dense CRF and save results to file

        Parameters
        ----------
        params : tuple
            Contains the settings for running dense CRF and saving segmentation results to file
        """
        img, featmap, category_num, id_, output_dir, should_overlay = params
        if 'DeepGlobe' in self.dataset:
            img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4))
            featmap = cv2.resize(featmap, (featmap.shape[0] // 4, featmap.shape[1] // 4),
                                 interpolation=cv2.INTER_NEAREST)
        if output_dir is not None:
            imgio.imsave(os.path.join(output_dir, "%s_img.png" % id_), np.uint8(img))
            if should_overlay:
                overlay_r = 0.75
                imgio.imsave(os.path.join(output_dir, "%s_output.png" % id_),
                             np.uint8((1 - overlay_r) * img + overlay_r *
                             self.label2rgb(np.argmax(featmap, axis=2), category_num)))
            else:
                imgio.imsave(os.path.join(output_dir, "%s_output.png" % id_),
                             np.uint8(self.label2rgb(np.argmax(featmap, axis=2), category_num)))
            imgio.imsave(os.path.join(output_dir, "%s_pred.png" % id_), np.uint8(self.label2rgb(np.argmax(featmap, axis=2),
                                                                                       category_num)))

    def eval_miou(self, data, id, gt, input, layer, dropout, val_category, saveimg_dir, should_overlay=False, is_eval=False):
        """Convert index labels to RGB colour images

        Parameters
        ----------
        data : Tensor
            Input images in batch, after resizing and normalizing
        id : Tensor
            Filenames in batch
        gt : Tensor
            GT segmentation in batch
        input : Layer
            Input layer of the network
        dropout : Dropout
            Dropout variable in the network
        val_category : str
            Name of validation set
        saveimg_dir : str
            Directory to save the debug images
        should_overlay : bool, optional
            Whether to overlay segmentation on original image
        is_eval : bool, optional
            Whether currently predicting on evaluation set

        Returns
        -------
        mIoU : float
            The mIoU of the complete evaluation set
        """
        if self.verbose:
            print('Evaluating mIoU on [%s] set' % val_category)
        intersect = np.zeros((self.num_classes))
        union = np.zeros((self.num_classes))
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        gt_count = np.zeros((self.num_classes))
        pred_count = np.zeros((self.num_classes))
        i = 0
        img_ids = []
        try:
            while True:
                # Read the input images, filenames, and GT segmentations in current batch
                img,id_,gt_ = self.sess.run([data,id,gt])
                # Generate predicted segmentation in current batch
                output_scale = self.sess.run(layer, feed_dict={input: img, dropout: 0.0})
                img_ids += list(id_)
                gt_ = gt_[:, :, :, :3]
                i += 1
                if self.verbose:
                    print('\tBatch #%d of %d' % (i, self.data_len[val_category] // self.batch_size + 1), end='')
                start_time = time.time()
                # Iterate through images in batch
                for j in range(img.shape[0]):
                    if not is_eval:
                        img_curr = np.uint8(img[j] + self.img_mean)
                        gt_curr = np.uint8(gt_[j])
                        pred_curr = output_scale[j]
                    else:
                        # Read original image
                        img_curr = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir, 'JPEGImages',
                                                id_[j].decode('utf-8') + '.jpg')), cv2.COLOR_RGB2BGR)
                        # Read GT segmentation
                        if self.dataset == 'VOC2012' or 'DeepGlobe' in self.dataset:
                            gt_curr = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir, 'SegmentationClassAug',
                                                          id_[j].decode('utf-8') + '.png')), cv2.COLOR_RGB2BGR)
                        elif self.dataset == 'ADP-morph':
                            gt_curr = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir, 'SegmentationClassAug', 'ADP-morph',
                                                              id_[j].decode('utf-8') + '.png')), cv2.COLOR_RGB2BGR)
                        elif self.dataset == 'ADP-func':
                            gt_curr = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir, 'SegmentationClassAug', 'ADP-func',
                                                              id_[j].decode('utf-8') + '.png')), cv2.COLOR_RGB2BGR)
                        # Read predicted segmentation
                        if 'DeepGlobe' not in self.dataset:
                            pred_curr = cv2.resize(output_scale[j], (gt_curr.shape[1], gt_curr.shape[0]))
                            img_curr = cv2.resize(img_curr, (gt_curr.shape[1], gt_curr.shape[0]))
                            # Apply dCRF
                            pred_curr = crf_inference(img_curr, self.model.crf_config_test, self.num_classes, pred_curr,
                                                      use_log=True)
                        else:
                            # Apply dCRF
                            pred_curr = crf_inference(np.uint8(img[j]), self.model.crf_config_test, self.num_classes,
                                                      output_scale[j], use_log=True)
                            pred_curr = cv2.resize(pred_curr, (gt_curr.shape[1], gt_curr.shape[0]))
                            img_curr = cv2.resize(img_curr, (gt_curr.shape[1], gt_curr.shape[0]))
                    # Evaluate predicted segmentation
                    if self.dataset == 'VOC2012':
                        pred_count += np.bincount(np.argmax(pred_curr, axis=-1).ravel(), minlength=self.num_classes)
                        for k in range(pred_curr.shape[2]):
                            gt_mask = gt_curr[:, :, 0] == k
                            pred_mask = np.argmax(pred_curr, axis=-1) == k
                            confusion_matrix[k, :] += np.bincount(np.argmax(pred_curr[gt_mask], axis=-1),
                                                                  minlength=self.num_classes)
                            gt_count[k] += np.sum(gt_mask)
                            intersect[k] += np.sum(gt_mask & pred_mask)
                            union[k] += np.sum(gt_mask | pred_mask)
                    elif self.dataset in ['ADP-morph', 'ADP-func'] or 'DeepGlobe' in self.dataset:
                        gt_r = gt_curr[:, :, 0]
                        gt_g = gt_curr[:, :, 1]
                        gt_b = gt_curr[:, :, 2]
                        pred_count += np.bincount(np.argmax(pred_curr, axis=-1).ravel(), minlength=self.num_classes)
                        for k, gt_colour in enumerate(self.label2rgb_colors):
                            gt_mask = (gt_r == gt_colour[0]) & (gt_g == gt_colour[1]) & (gt_b == gt_colour[2])
                            pred_mask = np.argmax(pred_curr, axis=-1) == k
                            confusion_matrix[k, :] += np.bincount(np.argmax(pred_curr[gt_mask], axis=-1), minlength=self.num_classes)
                            gt_count[k] += np.sum(gt_mask)
                            intersect[k] += np.sum(gt_mask & pred_mask)
                            union[k] += np.sum(gt_mask | pred_mask)
                    # Save image
                    if is_eval or (not is_eval and self.should_saveimg):
                        if not is_eval and self.dataset == 'VOC2012' and (j != 0):
                            continue
                        if 'ADP' in self.dataset and not is_eval:
                            img_curr = img_curr[:, :, ::-1]
                        params = (img_curr, pred_curr, self.num_classes, id_[j].decode(),
                                  saveimg_dir, should_overlay)
                        self.single_crf_metrics(params)
                if self.verbose:
                    print('\t\tElapsed time (s): %s' % (time.time() - start_time))
        except tf.errors.OutOfRangeError:
            print('Done evaluating mIoU on validation set')
        except Exception as e:
            print("Exception info:%s" % traceback.format_exc())
        # Evaluate mIoU and save to .xlsx file
        mIoU = np.mean(intersect / (union + 1e-7))
        precision = intersect / (gt_count + 1e-5)
        recall = intersect / (pred_count + 1e-5)
        if is_eval:
            df = pd.DataFrame({'Class': self.class_names + ['Mean'], 'IoU': list(intersect / (union + 1e-7)) + [mIoU],
                               'Precision': list(precision) + [np.mean(precision)],
                               'Recall': list(recall) + [np.mean(recall)]},
                              columns=['Class', 'IoU', 'Precision', 'Recall'])
            xlsx_path = os.path.join(self.eval_dir, 'metrics_' + self.dataset + '_' + val_category + '-' + self.seed_type + '.xlsx')
            df.to_excel(xlsx_path)
        # Save confusion matrix for all classes to .png file
        count_mat = np.transpose(np.matlib.repmat(gt_count, self.num_classes, 1))
        np.savetxt(os.path.join(self.eval_dir, 'confusion_' + self.dataset + '_' + val_category + '-' + self.seed_type + '.csv'),
                   confusion_matrix / count_mat)
        title = "Confusion matrix\n"
        xlabel = 'Prediction'  # "Labels"
        ylabel = 'Ground-Truth'  # "Labels"
        xticklabels = self.class_names
        yticklabels = self.class_names
        heatmap(confusion_matrix / count_mat, title, xlabel, ylabel, xticklabels, yticklabels, rot_angle=45)
        plt.savefig(os.path.join(self.eval_dir, 'confusion_' + self.dataset + '_' + val_category + '.png'), dpi=96, format='png', bbox_inches='tight')
        plt.close()
        # Save confusion matrix for only foreground classes to .png file
        title = "Confusion matrix\n"
        xlabel = 'Prediction'  # "Labels"
        ylabel = 'Ground-Truth'  # "Labels"
        if self.dataset == 'VOC2012' or self.dataset == 'ADP-morph':
            xticklabels = self.class_names[1:]
            yticklabels = self.class_names[1:]
            heatmap(confusion_matrix[1:, 1:] / (count_mat[1:, 1:] + 1e-7), title, xlabel, ylabel, xticklabels,
                    yticklabels, rot_angle=45)
        elif self.dataset == 'ADP-func':
            xticklabels = self.class_names[2:]
            yticklabels = self.class_names[2:]
            heatmap(confusion_matrix[2:, 2:] / (count_mat[2:, 2:] + 1e-7), title, xlabel, ylabel, xticklabels,
                    yticklabels, rot_angle=45)
        elif 'DeepGlobe' in self.dataset:
            xticklabels = self.class_names[:-1]
            yticklabels = self.class_names[:-1]
            heatmap(confusion_matrix[:-1, :-1] / (count_mat[:-1, :-1] + 1e-7), title, xlabel, ylabel, xticklabels,
                    yticklabels, rot_angle=45)
        plt.savefig(os.path.join(self.eval_dir, 'confusion_fore_' + self.dataset + '_' + val_category + '.png'), dpi=96,
                    format='png', bbox_inches='tight')
        plt.close()

        return mIoU