import os
import sys
import time
import math
import skimage
import skimage.io as imgio
import traceback
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from .crf import crf_inference

def single_crf_metrics(params):
    img,featmap,crf_config,category_num,id_,output_dir = params
    m = metrics_np(n_class=category_num)
    crf_output = crf_inference(img,crf_config,category_num,featmap,use_log=True)
    crf_pred = np.argmax(crf_output,axis=2)

    if output_dir is not None:
        img = img[:, :, [2, 1, 0]]
        imgio.imsave(os.path.join(output_dir,"%s_img.png"%id_),img/256.0)
        imgio.imsave(os.path.join(output_dir,"%s_output.png"%id_),dataset_dsrg.label2rgb(np.argmax(featmap,axis=2), category_num))
        # imgio.imsave(os.path.join(output_dir,"%s_label.png"%id_),dataset_dsrg.label2rgb(label[:,:,0]))
        imgio.imsave(os.path.join(output_dir,"%s_pred.png"%id_),dataset_dsrg.label2rgb(crf_pred, category_num))

    # m.update(label,crf_pred)
    return m.hist

class Predict():
    def __init__(self,config):
        self.config = config
        self.crf_config = config.get("crf",None)
        self.num_classes = self.config.get("num_classes",21)
        self.input_size = self.config.get("input_size",(240,240)) # (w,h)
        if self.input_size is not None:
            self.h,self.w = self.input_size
        else:
            self.h,self.w = None,None
   
        assert "sess" in self.config, "no session in config while using existing net"
        self.sess = self.config["sess"]
        assert "net" in self.config, "no network in config while using existing net"
        self.net = self.config["net"]
        assert "data" in self.config, "no dataset in config while using existing net"
        self.data = self.config["data"]

    def metrics_predict_tf_with_crf(self,category="val",multiprocess_num=100,crf_config=None,scales=[1.0],fixed_input_size=None,output_dir=None,use_max=False):
        print("predict config:")
        print("category:%s,\n multiprocess_num:%d,\n crf_config:%s,\n scales:%s,\n fixed_input_size:%s,\n output_dir:%s,\n use_max:%s" % (category,multiprocess_num,str(crf_config),str(scales),str(fixed_input_size),str(output_dir),str(use_max)))


        # pool = Pool(multiprocess_num)
        pool = Pool(None)
        i = 0
        m = metrics_np(n_class=self.num_classes)
        try:
            params = []
            while(True):
                img,id_ = self.sess.run([self.net["input"],self.net["id"]])
                origin_h,origin_w = img.shape[1:3]
                origin_img = img
                if fixed_input_size is not None:
                    img = nd.zoom(img,[1.0,fixed_input_size[0]/img.shape[1],fixed_input_size[1]/img.shape[2],1.0],order=1)
                output = np.zeros([1,origin_h,origin_w,self.num_classes])
                final_output = np.zeros([1,origin_h,origin_w,self.num_classes])
                for scale in scales:
                    scale_1 = 1.0/scale
                    img_scale = nd.zoom(img,[1.0,scale,scale,1.0],order=1)
                    output_scale = self.sess.run(self.net["rescale_output"],feed_dict={self.net["input"]:img_scale})
                    output_scale = nd.zoom(output_scale,[1.0,origin_h/output_scale.shape[1],origin_w/output_scale.shape[2],1.0],order=0)
                    output_scale_h,output_scale_w = output_scale.shape[1:3]
                    output_h_ = min(origin_h,output_scale_h)
                    output_w_ = min(origin_w,output_scale_w)
                    final_output[:,:output_h_,:output_w_,:] = output_scale[:,:output_h_,:output_w_,:]
                    if use_max is True:
                        output = np.max(np.stack([output,final_output],axis=4),axis=4)
                    else:
                        output += final_output

                        params.append((origin_img[0] + self.data.img_mean, output[0], crf_config,
                                       self.num_classes, id_[0].decode(), output_dir))
                if i >= 0: # % multiprocess_num == multiprocess_num -1:
                    print("start %d ..." % i)
                    single_crf_metrics(params[-1])
                    #print("params:%d" % len(params))
                    #print("params[0]:%d" % len(params[0]))
                    # if len(params) > 0:
                    #     ret = pool.map(single_crf_metrics,params)
                    #     for hist in ret:
                    #         m.update_hist(hist)
                    params = []
                i += 1
        except tf.errors.OutOfRangeError:
            # if len(params) > 0:
            #     ret = pool.map(single_crf_metrics,params)
            #     for hist in ret:
            #         m.update_hist(hist)
            print("output of range")
            # print("tf miou:%f" % m.get("miou"))
            # print("all metrics:%s" % str(m.get_all()))
        except Exception as e:
            print("exception info:%s" % traceback.format_exc())
        finally:
            pool.close()
            pool.join()
            print("finally")

    def metrics_debug_tf_with_crf(self,category="val",multiprocess_num=100,crf_config=None,scales=[1.0],fixed_input_size=None,output_dir=None,use_max=False):
        print("debug config:")
        print("category:%s,\n multiprocess_num:%d,\n crf_config:%s,\n scales:%s,\n fixed_input_size:%s,\n output_dir:%s,\n use_max:%s" % (category,multiprocess_num,str(crf_config),str(scales),str(fixed_input_size),str(output_dir),str(use_max)))
        # pool = Pool(multiprocess_num)
        pool = Pool(None)
        i = 0
        try:
            params = []
            while(True):
                img,id_ = self.sess.run([self.net["input"],self.net["id"]])
                origin_h,origin_w = img.shape[1:3]
                origin_img = img
                if fixed_input_size is not None:
                    img = nd.zoom(img,[1.0,fixed_input_size[0]/img.shape[1],fixed_input_size[1]/img.shape[2],1.0],order=1)
                output = np.zeros([1,origin_h,origin_w,self.num_classes])
                final_output = np.zeros([1,origin_h,origin_w,self.num_classes])
                output_scale = self.sess.run(self.net["rescale_output"],feed_dict={self.net["input"]:img})
                output_scale = nd.zoom(output_scale,[1.0,origin_h/output_scale.shape[1],origin_w/output_scale.shape[2],1.0],order=0)
                output_scale_h,output_scale_w = output_scale.shape[1:3]
                output_h_ = min(origin_h,output_scale_h)
                output_w_ = min(origin_w,output_scale_w)
                final_output[:,:output_h_,:output_w_,:] = output_scale[:,:output_h_,:output_w_,:]

                should_debug_plot = True
                if should_debug_plot:
                    ## Check losses
                    ### Get tensors
                    fc8_t = self.net["fc8"]
                    softmax_t = self.net["fc8-softmax"]
                    oldcues_t = self.net["cues"]
                    cues_t = self.net["new_cues"]
                    crf_t = self.net["crf"]
                    count_bg_t = tf.reduce_sum(cues_t[:, :, :, 0:1], axis=(1, 2, 3), keepdims=True)
                    loss_bg_px_t = -(cues_t[:, :, :, 0] * tf.log(softmax_t[:, :, :, 0])) / count_bg_t
                    count_fg_t = tf.reduce_sum(cues_t[:, :, :, 1:], axis=(1, 2, 3), keepdims=True)
                    loss_fg_px_t = -(cues_t[:, :, :, 1:] * tf.log(softmax_t[:, :, :, 1:])) / count_fg_t
                    loss_constrain_t = tf.exp(crf_t) * tf.log(tf.exp(crf_t) / (softmax_t + 1e-8) + 1e-8)

                    ### Get values
                    fc8_v, softmax_v, oldcues_v, cues_v, crf_v, loss_bg_px_v, loss_fg_px_v, loss_constrain_v = self.sess.run(
                        [fc8_t, softmax_t, oldcues_t, cues_t, crf_t, loss_bg_px_t, loss_fg_px_t, loss_constrain_t])
                    softmax_argmax_v = np.argmax(softmax_v[0], axis=-1)
                    cues_argmax_v = np.argmax(cues_v[0], axis=-1)
                    ### Visualize
                    class_ind = 2  # 9: C.L, 2: E.M.O, 10: H.E
                    class_name = 'E.M.O'  # C.L, E.M.O, H.E
                    plt.figure(1)
                    plt.subplot(4, 2, 1)
                    plt.imshow(np.argmax(oldcues_v[0], axis=-1), interpolation='none')
                    plt.title('Old cues')
                    plt.subplot(4, 2, 2)
                    plt.imshow(cues_argmax_v, interpolation='none')
                    plt.title('New cues')
                    plt.subplot(4, 2, 3)
                    plt.imshow(softmax_argmax_v, interpolation='none')
                    plt.title('Max-confidence Softmax')
                    plt.subplot(4, 2, 4)
                    plt.imshow(np.argmax(crf_v[0], axis=-1), interpolation='none')
                    plt.title('Max-confidence CRF')
                    plt.subplot(4, 2, 5)
                    plt.imshow(np.log(softmax_v[0, :, :, class_ind]), interpolation='none')
                    plt.title(class_name + ' Log-Softmax')
                    plt.subplot(4, 2, 6)
                    plt.imshow(crf_v[0, :, :, class_ind + 1])
                    plt.title(class_name + ' CRF')
                    plt.subplot(4, 2, 7)
                    plt.imshow(loss_fg_px_v[0, :, :, class_ind])
                    plt.title(class_name + ' Seed Loss')
                    plt.subplot(4, 2, 8)
                    plt.imshow(loss_constrain_v[0, :, :, class_ind], interpolation='none')
                    plt.title(class_name + ' Constrain Loss')

                    plt.figure(2)
                    bg_loss_reformatted = np.expand_dims(np.expand_dims(np.squeeze(loss_bg_px_v), axis=0), axis=3)
                    loss_seed_px_v = np.concatenate((bg_loss_reformatted, loss_fg_px_v), axis=3)
                    loss_seed_px_v = np.sum(loss_seed_px_v, axis=3)
                    plt.subplot(3, 1, 1)
                    plt.imshow(loss_seed_px_v[0], interpolation='none')
                    plt.title('Total Seed Loss')
                    plt.subplot(3, 1, 2)
                    loss_constrain_px_v = np.sum(loss_constrain_v, axis=3)
                    plt.imshow(loss_constrain_px_v[0], interpolation='none')
                    plt.title('Total Constrain Loss')
                    loss_total_px_v = np.concatenate((np.expand_dims(loss_seed_px_v, axis=3), loss_constrain_v), axis=3)
                    loss_total_px_v = np.sum(loss_total_px_v, axis=3)
                    plt.subplot(3, 1, 3)
                    plt.imshow(loss_total_px_v[0], interpolation='none')
                    plt.title('Total Loss')

                    plt.figure(3)
                    plt.subplot(2, 1, 1)
                    plt.imshow(fc8_v[0, :, :, class_ind + 1], interpolation='none')
                    plt.title(class_name + ' fc8')
                    plt.subplot(2, 1, 2)
                    plt.imshow(crf_v[0, :, :, class_ind + 1], interpolation='none')
                    plt.title(class_name + ' CRF')
                    plt.show()

                if use_max is True:
                    output = np.max(np.stack([output,final_output],axis=4),axis=4)
                else:
                    output += final_output

                    params.append((origin_img[0] + self.data.img_mean, output[0], crf_config,
                                   self.num_classes, id_[0].decode(), output_dir))
                if i >= 0: # % multiprocess_num == multiprocess_num -1:
                    print("start %d ..." % i)
                    single_crf_metrics(params[-1])
                    params = []
                i += 1
        except tf.errors.OutOfRangeError:
            print("output of range")
        except Exception as e:
            print("exception info:%s" % traceback.format_exc())
        finally:
            pool.close()
            pool.join()
            print("finally")

# Originally written by wkentaro for the numpy version
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
class metrics_np():
    def __init__(self,n_class=1,hist=None):
        if hist is None:
            self.hist = np.zeros((n_class,n_class))
        else:
            self.hist = hist
        self.n_class = n_class

    def _fast_hist(self,label_true,label_pred,n_class):
        mask = (label_true>=0) & (label_true<n_class) # to ignore void label
        self.hist = np.bincount( n_class * label_true[mask].astype(int)+label_pred[mask],minlength=n_class**2).reshape(n_class,n_class)
        return self.hist

    def update(self,x,y):
        self.hist += self._fast_hist(x.flatten(),y.flatten(),self.n_class)

    def update_hist(self,hist):
        self.hist += hist

    def get(self,kind="miou"):
        if kind == "accu":
            return np.diag(self.hist).sum() / (self.hist.sum()+1e-3) # total pixel accuracy
        elif kind == "precision":
            return np.diag(self.hist) / (self.hist.sum(axis=0)+1e-3) 
        elif kind == "recall":
            return np.diag(self.hist) / (self.hist.sum(axis=1)+1e-3) 
        elif kind in ["freq","fiou","iou","miou"]:
            iou = np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0) - np.diag(self.hist)+1e-3)
            if kind == "iou": return iou
            miou = np.nanmean(iou)
            if kind == "miou": return miou

            freq = self.hist.sum(axis=1) / (self.hist.sum()+1e-3) # the frequency for each categorys
            if kind == "freq": return freq
            else: return (freq[freq>0]*iou[freq>0]).sum()
        elif kind in ["dice","mdice"]:
            dice = 2*np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0)+1e-3)
            if kind == "dice": return dice
            else: return np.nanmean(dice)
        return None

    def get_all(self):
     metrics = {}
     metrics["accu"] = np.diag(self.hist).sum() / (self.hist.sum()+1e-3) # total pixel accuracy
     metrics["precision"] = np.diag(self.hist) / (self.hist.sum(axis=0)+1e-3) # pixel accuracys for each category, np.nan represent the corresponding category not exists
     metrics["recall"] = np.diag(self.hist) / (self.hist.sum(axis=1)+1e-3) # pixel accuracys for each category, np.nan represent the corresponding category not exists
     metrics["iou"] = np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0) - np.diag(self.hist)+1e-3)
     metrics["miou"] = np.nanmean(metrics["iou"])
     metrics["freq"] = self.hist.sum(axis=1) / (self.hist.sum()+1e-3) # the frequency for each categorys
     metrics["fiou"] = (metrics["freq"][metrics["freq"]>0]*metrics["iou"][metrics["freq"]>0]).sum()
     metrics["dices"] = 2*np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0)+1e-3)
     metrics["mdice"] = np.nanmean(metrics["dices"])
 
     return metrics

