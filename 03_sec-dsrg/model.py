import os
import time
import pickle
import cv2
import numpy.matlib
import tensorflow as tf
import skimage.color as imgco
import skimage.io as imgio
import multiprocessing
import pandas as pd
import traceback

from utilities import *
from lib.crf import crf_inference

from DSRG import DSRG
from SEC import SEC

MODEL_WSSS_ROOT = '../database/models_wsss'

class Model():
    def __init__(self, args):
        self.method = args.method
        self.dataset = args.dataset
        self.phase = 'predict'
        self.seed_type = args.seed
        if self.dataset in ['ADP-morph', 'ADP-func']:
            self.setname = args.setname
            self.sess_id = self.dataset + '_' + self.setname + '_' + self.seed_type
        else:
            self.sess_id = self.dataset + '_' + self.seed_type
        self.h, self.w = (321, 321)
        self.seed_size = 41
        self.batch_size = args.batchsize
        self.should_saveimg = args.saveimg

        self.accum_num = 1
        self.pool = multiprocessing.Pool()

        self.saver = {}

        self.l2loss = {"total": 0}

        # paths
        self.save_dir = os.path.join(MODEL_WSSS_ROOT, self.method, self.dataset + '_' + self.seed_type)
        self.out_dir = os.path.join('out', self.method, self.sess_id)
        self.eval_dir = os.path.join('eval', self.method, self.sess_id)
        for pth in [self.out_dir, self.eval_dir]:
            if not os.path.exists(pth):
                os.makedirs(pth)

        if self.method == 'DSRG':
            self.init_model_path = os.path.join(MODEL_WSSS_ROOT, self.method, 'vgg16_deeplab_aspp.npy')
        elif self.method == 'SEC':
            self.init_model_path = os.path.join(MODEL_WSSS_ROOT, self.method, 'init.npy')

        database_dir = os.path.join(os.path.dirname(os.getcwd()), 'database')

        if self.dataset == 'ADP-morph':
            self.num_classes = 29
            self.img_mean = np.array([208.8502, 163.2828, 207.1458])
            self.dataset_dir = os.path.join(database_dir, 'ADPdevkit', 'ADPRelease1')
            self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list.txt')
            self.run_categories = [self.setname] # self.run_categories \in ['tuning', 'segtest']
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
            self.dataset_dir = os.path.join(database_dir, 'ADPdevkit', 'ADPRelease1')
            self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list.txt')
            self.run_categories = [self.setname]  # self.run_categories = ['tuning', 'segtest']
            self.class_names = ['Background', 'Other', 'G.O', 'G.N', 'T']
            self.label2rgb_colors = np.array([(255, 255, 255), (3, 155, 229), (0, 0, 128), (0, 128, 0), (173, 20, 87)])
        elif self.dataset == 'VOC2012':
            self.num_classes = 21
            self.img_mean = np.array([104.00698793, 116.66876762, 122.67891434])
            self.dataset_dir = os.path.join(database_dir, 'VOCdevkit', 'VOC2012')
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
            self.dataset_dir = os.path.join(database_dir, 'DGdevkit')
            if self.dataset == 'DeepGlobe_train75':
                self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list_train75.txt')
            elif self.dataset == 'DeepGlobe_train37.5':
                self.input_path = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation', 'input_list_train37.5.txt')
            self.run_categories = ['test']
            # if 'DeepGlobe2' in self.dataset:
            #     self.run_categories = ['test']
            # else:
            #     self.run_categories = ['val']
            self.class_names = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren']
            self.label2rgb_colors = np.array([(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255),
                        (255, 255, 255)])
        self.img_mean = np.expand_dims(np.expand_dims(self.img_mean, axis=0), axis=1)

        # For dataset
        self.ignore_label = 255

        self.run_categories = ['train'] + self.run_categories
        self.data_f, self.data_len = self.get_data_f(self.run_categories)
    def load(self):
        if self.method == 'DSRG':
            self.model = DSRG({'dataset': self.dataset, 'img_size': self.h, 'num_classes': self.num_classes,
                               'batch_size': self.batch_size, 'phase': self.phase, 'img_mean': self.img_mean,
                               'seed_size': self.seed_size, 'pool': self.pool, 'init_model_path': self.init_model_path})
        elif self.method == 'SEC':
            self.model = SEC({'dataset': self.dataset, 'img_size': self.h, 'num_classes': self.num_classes,
                              'batch_size': self.batch_size, 'phase': self.phase, 'img_mean': self.img_mean,
                              'seed_size': self.seed_size, 'pool': self.pool, 'init_model_path': self.init_model_path})
    def get_data_f(self, phases):
        # cues_labels = [self.cues_data[x] for i, x in enumerate(self.cues_data.keys()) if '_labels' in x]
        # [i for i,x in enumerate(cues_labels) if 32 in x]
        cues_root = os.path.join(os.path.dirname(os.getcwd()), '01_weak_cues')
        data_f = {}
        data_len = {}
        for phase in phases:
            data_f[phase] = {"img": [], "label": [], "id": [], "id_for_slice": []}
            data_len[phase] = 0
            if phase in ['train']:
                if 'ADP' in self.dataset:
                    cues_path = os.path.join(cues_root, 'cues_train', 'ADP_' + self.seed_type,
                                             self.dataset.split('-')[-1], 'localization_cues.pickle')
                else:
                    cues_path = os.path.join(cues_root, 'cues_train', self.dataset + '_' + self.seed_type,
                                             'localization_cues.pickle')
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
        print("len:%s" % str(data_len))
        return data_f,data_len

    def next_batch(self,category=None,max_epochs=-1):
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
                #"gt_f":self.data_f[category]["gt"],
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
            # dataset = dataset.shuffle(self.data_len[category])
            dataset = dataset.map(m_val)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            img, label, id = iterator.get_next()

            return img, label, id, iterator

    def image_preprocess(self, img, label=None):
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
                # label -= self.ignore_label
                # label = tf.squeeze(label, squeeze_dims=[0])
                # label += self.ignore_label
                # label = tf.expand_dims(label, axis=0)
                # label = tf.image.resize_nearest_neighbor(label, (self.h, self.w))
                # label = tf.squeeze(label, axis=0)

            r, g, b = tf.split(axis=2, num_or_size_splits=3, value=img)
            img = tf.cast(tf.concat([b, g, r], 2), dtype=tf.float32)
            img -= self.img_mean

            return img, label

    def label2rgb(self, label, category_num, colors=[], ignore_label=128, ignore_color=(255,255,255)):
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
        if 'final-0.index' in os.listdir(self.save_dir):
            return os.path.join(self.save_dir, 'final-0')
        all_checkpoint_inds = [int(x.split('.')[0].split('-')[-1]) for x in os.listdir(self.save_dir) if
                                 'epoch-' in x and '.index' in x]
        if len(all_checkpoint_inds) == 0:
            return None
        latest_checkpoint = os.path.join(self.save_dir, 'epoch-' + str(max(all_checkpoint_inds)))
        return latest_checkpoint

    def restore_from_model(self,saver,model_path,checkpoint=False):
        assert self.sess is not None
        if checkpoint is True:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(self.sess, model_path)

    def predict(self):
        tf.reset_default_graph()
        # self.sess = tf.Session()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
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
            print('Loading model from previous checkpoint %s' % latest_ckpt)
            self.restore_from_model(self.saver, latest_ckpt, checkpoint=False)

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
                print('mIoU [%s] = %f' % (val_category, miou_val))

    def single_crf_metrics(self, params):
        img, featmap, category_num, id_, output_dir, should_overlay = params
        if 'DeepGlobe' in self.dataset:
            img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4))
            featmap = cv2.resize(featmap, (featmap.shape[0] // 4, featmap.shape[1] // 4),
                                 interpolation=cv2.INTER_NEAREST)
        if output_dir is not None:
            imgio.imsave(os.path.join(output_dir, "%s_img.png" % id_), img / 256.0)
            if should_overlay:
                overlay_r = 0.75
                imgio.imsave(os.path.join(output_dir, "%s_output.png" % id_),
                             (1-overlay_r) * img / 256.0 + overlay_r *
                             self.label2rgb(np.argmax(featmap, axis=2), category_num) / 256.0)
            else:
                imgio.imsave(os.path.join(output_dir, "%s_output.png" % id_),
                             self.label2rgb(np.argmax(featmap, axis=2), category_num))
            imgio.imsave(os.path.join(output_dir, "%s_pred.png" % id_), self.label2rgb(np.argmax(featmap, axis=2),
                                                                                       category_num))

    def eval_miou(self, data, id, gt, input, layer, dropout, val_category, saveimg_dir, should_overlay=False, is_eval=False):
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
                img,id_,gt_ = self.sess.run([data,id,gt])
                output_scale = self.sess.run(layer,feed_dict={input:img, dropout:0.0})
                img_ids += list(id_)
                gt_ = gt_[:, :, :, :3]
                i += 1
                print('\tBatch #%d of %d' % (i, self.data_len[val_category] // self.batch_size + 1), end='')
                start_time = time.time()
                for j in range(img.shape[0]):
                    if not is_eval:
                        img_curr = np.uint8(img[j] + self.img_mean)
                        gt_curr = np.uint8(gt_[j])
                        pred_curr = output_scale[j]
                    else:
                        img_curr = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir, 'JPEGImages',
                                                id_[j].decode('utf-8') + '.jpg')), cv2.COLOR_RGB2BGR)
                        if self.dataset == 'VOC2012' or 'DeepGlobe' in self.dataset:
                            gt_curr = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir, 'SegmentationClassAug',
                                                          id_[j].decode('utf-8') + '.png')), cv2.COLOR_RGB2BGR)
                        elif self.dataset == 'ADP-morph':
                            gt_curr = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir, 'SegmentationClassAug', 'ADP-morph',
                                                              id_[j].decode('utf-8') + '.png')), cv2.COLOR_RGB2BGR)
                        elif self.dataset == 'ADP-func':
                            gt_curr = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_dir, 'SegmentationClassAug', 'ADP-func',
                                                              id_[j].decode('utf-8') + '.png')), cv2.COLOR_RGB2BGR)
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
                    if self.dataset == 'VOC2012':
                        pred_count += np.bincount(np.argmax(pred_curr, axis=-1).ravel(), minlength=self.num_classes)
                        for k in range(pred_curr.shape[2]):
                            gt_mask = gt_curr[:, :, 0] == k
                            pred_mask = np.argmax(pred_curr, axis=-1) == k
                            confusion_matrix[k, :] += np.bincount(np.argmax(pred_curr[gt_mask], axis=-1),
                                                                  minlength=self.num_classes)
                            gt_count[k] += np.sum(gt_mask)
                            #
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
                            #
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
                print('\t\tElapsed time (s): %s' % (time.time() - start_time))
        except tf.errors.OutOfRangeError:
            print('Done evaluating mIoU on validation set')
        except Exception as e:
            print("exception info:%s" % traceback.format_exc())
        mIoU = np.mean(intersect / (union + 1e-7))
        if is_eval:
            df = pd.DataFrame({'Class': self.class_names + ['Mean'], 'IoU': list(intersect / (union + 1e-7)) + [mIoU]},
                              columns=['Class', 'IoU'])
            xlsx_path = os.path.join(self.eval_dir, 'metrics_' + self.dataset + '_' + val_category + '-' + self.seed_type + '.xlsx')
            df.to_excel(xlsx_path)
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