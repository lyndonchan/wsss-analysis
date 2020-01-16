import os
import cv2
import keras
from keras.models import model_from_json
from keras import optimizers
import numpy as np
import keras.backend as K
import scipy
import scipy.io as io
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import matplotlib.pyplot as plt

def build_model(model_dir, model_name):
    # Load architecture from json
    model_json_path = os.path.join(model_dir, model_name + '.json')
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights from h5
    model_h5_path = os.path.join(model_dir, model_name + '.h5')
    model.load_weights(model_h5_path)

    # Evaluate model
    opt = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    return model

def load_thresholds(model_dir, model_name):
    thresh_path = os.path.join(model_dir, model_name + '.mat')
    tmp = io.loadmat(thresh_path)
    return tmp.get('optimalScoreThresh')

def load_classes(dataset):
    if dataset == 'VOC2012':
        class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor']  # 20 classes
        seg_class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor']
    elif 'DeepGlobe' in dataset:
        class_names = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren']
        seg_class_names = class_names
    return class_names, seg_class_names

def load_colours(dataset):
    if dataset == 'VOC2012':
        return np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                          (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                          (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                          (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                          (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                          (0, 64, 128)])  # using palette for pascal voc
    elif 'DeepGlobe' in dataset:
        return np.array([(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255),
                        (255, 255, 255), (0, 0, 0)])
    elif dataset == 'CityScapes':
        return np.array([(0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160),
                         (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153), (180, 165, 180),
                         (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
                         (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0),
                         (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230),
                         (119, 11, 32), (0, 0, 142)])

def get_colours(segset):
    if segset == 'ADP-morph':
        return np.array([(255, 255, 255), (0, 0, 128), (0, 128, 0), (255, 165, 0), (255, 192, 203),
                         (255, 0, 0), (173, 20, 87), (176, 141, 105), (3, 155, 229),
                         (158, 105, 175), (216, 27, 96), (244, 81, 30), (124, 179, 66),
                         (142, 36, 255), (240, 147, 0), (204, 25, 165), (121, 85, 72),
                         (142, 36, 170), (179, 157, 219), (121, 134, 203), (97, 97, 97),
                         (167, 155, 142), (228, 196, 136), (213, 0, 0), (4, 58, 236),
                         (0, 150, 136), (228, 196, 65), (239, 108, 0), (74, 21, 209)])
    elif segset == 'ADP-func':
        return np.array([(255, 255, 255), (3, 155, 229), (0, 0, 128), (0, 128, 0), (173, 20, 87)])
    elif segset == 'VOC2012':
        return np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                         (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                         (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                         (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                         (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                         (0, 64, 128)])  # using palette for pascal voc
    elif 'DeepGlobe' in segset:
        return np.array([(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255),
                         (255, 255, 255), (0, 0, 0)])

def normalize(dataset, x):
    if dataset == 'VOC2012':
        x[:, :, 0] -= 104
        x[:, :, 1] -= 117
        x[:, :, 2] -= 123
        return x / 255
    elif dataset == 'ADP':
        return (x - 193.09203) / 56.450138
    elif 'DeepGlobe' in dataset:
        return x / 255

def read_batch(img_dir, batch_names, batch_sz, sz, dataset):
    img_batch = np.empty((batch_sz, sz[0], sz[1], 3), dtype='uint8')
    for i in range(batch_sz):
        if 'PNGImages' in img_dir:
            gt_name = os.path.splitext(batch_names[i])[0] + '.png'
        else:
            gt_name = batch_names[i]
        tmp = cv2.cvtColor(cv2.imread(os.path.join(img_dir, gt_name)), cv2.COLOR_RGB2BGR)
        img_batch[i] = cv2.resize(tmp, (sz[0], sz[1]))
    img_batch_norm = normalize(dataset, img_batch)
    return img_batch_norm, img_batch

def get_grad_cam_weights(input_model, dummy_image, should_normalize=True):
    def find_final_layer(model):
        for iter_layer, layer in reversed(list(enumerate(model.layers))):
            if type(layer) == type(layer) == keras.layers.convolutional.Conv2D:
                return model.layers[iter_layer + 1].name
        raise Exception('Could not find the final layer in provided HistoNet')
    final_layer = find_final_layer(input_model)
    conv_output = input_model.get_layer(final_layer).output  # activation_7
    num_classes = input_model.output_shape[1]
    num_feats = int(conv_output.shape[-1])
    weights = np.zeros((num_feats, num_classes))

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    for iter_class in range(input_model.output_shape[1]):
        y_c = input_model.layers[-2].output[0, iter_class]
        if should_normalize:
            grad = normalize(K.gradients(y_c, conv_output)[0])
        else:
            grad = K.gradients(y_c, conv_output)[0]
        grad_func = K.function([input_model.layers[0].input, K.learning_phase()], [conv_output, grad])
        conv_val, grad_val = grad_func([dummy_image, 0])
        conv_val, grad_val = conv_val[0], grad_val[0]
        weights[:, iter_class] = np.mean(grad_val, axis=(0, 1))
    return weights, final_layer


def grad_cam(input_model, weights, images, is_pass_threshold, final_layer, conf_scores, orig_sz=[224, 224],
             should_upsample=False, batch_size=16):
    conv_output = input_model.get_layer(final_layer).output  # activation_7
    conv_func = K.function([input_model.layers[0].input], [conv_output])
    conv_val = conv_func([images])
    conv_val = conv_val[0]

    cams = np.einsum('ijkl,lm->ijkm', conv_val, weights)
    if should_upsample:
        old_cams = cams[:]
        cams = np.zeros((old_cams.shape[0], orig_sz[0], orig_sz[1], old_cams.shape[-1]))
        for i in range(cams.shape[0]):
            for j in range(cams.shape[-1]):
                # cams[i, :, :, j] = cv2.resize(old_cams[i, :, :, j], (orig_sz[0], orig_sz[1]))
                cams[i, :, :, j] = np.maximum(cv2.resize(old_cams[i, :, :, j], (orig_sz[0], orig_sz[1])), 0)
    should_normalize = True
    if should_normalize:
        cams = cams / np.maximum(np.max(cams, axis=(1, 2, 3), keepdims=True), 1e-7)
        cams = cams * np.expand_dims(np.expand_dims(conf_scores * is_pass_threshold, axis=1), axis=2)
    else:
        cams = cams * np.expand_dims(np.expand_dims(is_pass_threshold, axis=1), axis=2)
    return cams


def split_by_httclass(H, all_classes, morph_classes, func_classes):
    morph_all_inds = [i for i, x in enumerate(all_classes) if x in morph_classes]
    func_all_inds = [i for i, x in enumerate(all_classes) if x in func_classes]
    return H[:, morph_all_inds], H[:, func_all_inds]

def modify_by_htt(gradcam, images, classes, gradcam_adipose=None):
    if gradcam_adipose is None:
        htt_class = 'morph'
    else:
        htt_class = 'func'
    if htt_class == 'morph':
        background_max = 0.75
        background_exception_classes = ['A.W', 'A.B', 'A.M']
    elif htt_class == 'func':
        background_max = 0.75
        other_tissue_mult = 0.05
        background_exception_classes = ['G.O', 'G.N', 'T']
        if gradcam_adipose is None:
            raise Exception('You must feed in adipose heatmap for functional type')
        other_ind = classes.index('Other')
    background_ind = classes.index('Background')

    # Get background class prediction
    mean_img = np.mean(images, axis=-1)
    sigmoid_input = 4 * (mean_img - 240)
    background_gradcam = background_max * scipy.special.expit(sigmoid_input)
    background_exception_cur_inds = [i for i, x in enumerate(classes) if x in background_exception_classes]
    for iter_input_image in range(background_gradcam.shape[0]):
        background_gradcam[iter_input_image] = scipy.ndimage.gaussian_filter(background_gradcam[iter_input_image],
                                                                             sigma=2)
    if background_gradcam.shape[1] != gradcam.shape[2] or background_gradcam.shape[2] != gradcam.shape[3]:
        old_bg = background_gradcam[:]
        background_gradcam = np.zeros((old_bg.shape[0], gradcam.shape[2], gradcam.shape[3]))
        for i in range(background_gradcam.shape[0]):
            background_gradcam[i] = cv2.resize(old_bg[i], (gradcam.shape[2], gradcam.shape[3]))
    background_gradcam -= np.max(gradcam[:, background_exception_cur_inds], axis=1)
    gradcam[:, background_ind] = background_gradcam

    # Get other tissue class prediction
    if htt_class == 'func':
        other_moh = np.max(gradcam, axis=1)
        other_gradcam = np.expand_dims(other_tissue_mult * (1 - other_moh), axis=1)
        other_gradcam = np.max(np.concatenate((other_gradcam, gradcam_adipose), axis=1), axis=1)
        # other_gradcam = np.clip(other_gradcam, 0, 1)
        gradcam[:, other_ind] = other_gradcam
    return gradcam


def get_cs_gradcam(gradcam, classes, htt_class):
    if htt_class in ['func', 'glas']:
        other_ind = classes.index('Other')
    # Find max difference value, ind map
    gradcam_sorted = np.sort(gradcam, axis=1)
    maxdiff = gradcam_sorted[:, -1] - gradcam_sorted[:, -2]
    maxind = np.argmax(gradcam, axis=1)
    # Find CS-Grad-CAM
    cs_gradcam = np.transpose(np.tile(np.expand_dims(maxdiff, axis=-1), gradcam.shape[1]), (0, 3, 1, 2))
    for iter_class in range(gradcam.shape[1]):
        if not (htt_class in ['func', 'glas'] and iter_class == other_ind):
            cs_gradcam[:, iter_class] *= (maxind == iter_class)
        else:
            cs_gradcam[:, iter_class] = gradcam[:, iter_class]
    return cs_gradcam


def dcrf_process(probs, images, config):
    gauss_sxy, gauss_compat, bilat_sxy, bilat_srgb, bilat_compat, n_infer = config

    # Set up variable sizes
    num_input_images = probs.shape[0]
    num_classes = probs.shape[1]
    size = images.shape[1:3]
    crf = np.zeros((num_input_images, num_classes, size[0], size[1]))
    for iter_input_image in range(num_input_images):
        pass_class_inds = np.where(np.sum(np.sum(probs[iter_input_image], axis=1), axis=1) > 0)
        # Set up dense CRF 2D
        d = dcrf.DenseCRF2D(size[1], size[0], len(pass_class_inds[0]))
        if len(pass_class_inds[0]) > 0:
            cur_probs = probs[iter_input_image, pass_class_inds[0]]
            # Unary energy
            U = np.ascontiguousarray(unary_from_softmax(cur_probs))
            d.setUnaryEnergy(U)
            # Penalize small, isolated segments
            # (sxy are PosXStd, PosYStd)
            d.addPairwiseGaussian(sxy=gauss_sxy, compat=gauss_compat)
            # Incorporate local colour-dependent features
            # (sxy are Bi_X_Std and Bi_Y_Std,
            #  srgb are Bi_R_Std, Bi_G_Std, Bi_B_Std)
            d.addPairwiseBilateral(sxy=bilat_sxy, srgb=bilat_srgb, rgbim=np.uint8(images[iter_input_image]),
                                   compat=bilat_compat)
            # Do inference
            Q = d.inference(n_infer)
            crf[iter_input_image, pass_class_inds] = np.array(Q).reshape((len(pass_class_inds[0]), size[0], size[1]))
    maxconf_crf = np.argmax(crf, axis=1)
    return maxconf_crf


def maxconf_class_as_colour(maxconf_crf, colours, size):
    num_input_images = maxconf_crf.shape[0]
    Y = np.zeros((num_input_images, size[0], size[1], 3), dtype='uint8')
    for iter_input_image in range(num_input_images):
        for iter_class in range(colours.shape[0]):
            Y[iter_input_image, maxconf_crf[iter_input_image] == iter_class] = np.array(colours[iter_class])
    return Y

def resize_stack(stack, size):
    old_stack = stack[:]
    stack = np.zeros((stack.shape[0], stack.shape[1], size[0], size[1]))
    for i in range(stack.shape[0]):
        for j in range(stack.shape[1]):
            stack[i, j] = cv2.resize(old_stack[i, j], (size[0], size[1]))
    return stack

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, rot_angle=0):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='afmhot', vmin=0.0, vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    # ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    if rot_angle == 45:
        ax.set_xticks(np.arange(AUC.shape[1]), minor=False)
        ax.set_xticklabels(xticklabels, minor=False, rotation=rot_angle, horizontalalignment='left')
    else:
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(xticklabels, minor=False, rotation=rot_angle)
    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    # plt.title(title, y=1.08, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    axis_offset = -0.012*AUC.shape[0] + 1.436
    ax.xaxis.set_label_coords(.5, axis_offset)

    # Remove last blank column
    # plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    ax.axis('equal')
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    # plt.colorbar(c)

    # Add text in each cell
    cell_font = 10 # math.ceil(AUC.shape[1] * 10 / 28)
    show_values(c, fontsize=cell_font)

    # Proper orientation (origin at the top left instead of bottom left)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    ax.axis('tight')
    fig_len = 40 / 28 * AUC.shape[0]

    fig.set_size_inches(cm2inch(fig_len, fig_len))