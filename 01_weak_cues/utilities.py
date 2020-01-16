import keras
import keras.backend as K
import os
import cv2
import numpy as np
import scipy

def makedir_if_nexist(dir_list):
    for cur_dir in dir_list:
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

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

def find_final_layer(model):
    for iter_layer, layer in reversed(list(enumerate(model.layers))):
        if type(layer) == type(layer) == keras.layers.convolutional.Conv2D:
            return model.layers[iter_layer+1].name
    raise Exception('Could not find the final layer in provided HistoNet')

def get_grad_cam_weights(input_model, final_layer, dummy_image, should_normalize=True):
    conv_output = input_model.get_layer(final_layer).output # activation_7
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
    return weights

def grad_cam(input_model, weights, images, is_pass_threshold, final_layer, keep_inds, orig_sz=[224, 224], should_upsample=False):
    conv_output = input_model.get_layer(final_layer).output  # activation_7
    conv_func = K.function([input_model.layers[0].input], [conv_output])
    conv_val = conv_func([images])
    conv_val = conv_val[0]
    cams = np.maximum(np.einsum('ijkl,lm->ijkm', conv_val, weights), 0)
    cams = cams[:, :, :, keep_inds]
    if should_upsample:
        old_cams = cams[:]
        cams = np.zeros((old_cams.shape[0], orig_sz[0], orig_sz[1], old_cams.shape[-1]))
        for i in range(cams.shape[0]):
            for j in range(cams.shape[-1]):
                cams[i, :, :, j] = cv2.resize(cams[i, :, :, j], (orig_sz[0], orig_sz[1]))
    cams_thresh = cams * np.expand_dims(np.expand_dims(is_pass_threshold, axis=1), axis=2)
    return cams_thresh

def read_batch(img_dir, batch_names, batch_sz, sz, img_mean=[193.09203, 193.09203, 193.02903],
               img_std=[56.450138, 56.450138, 56.450138]):
    img_mean = np.float64(img_mean)
    img_std = np.float64(img_std)
    img_batch = np.empty((batch_sz, sz[0], sz[1], 3), dtype='uint8')
    for i in range(batch_sz):
        tmp = cv2.cvtColor(cv2.imread(os.path.join(img_dir, batch_names[i])), cv2.COLOR_RGB2BGR)
        if tmp.shape[:2] != sz:
            img_batch[i] = cv2.resize(tmp, (sz[0], sz[1]))
    img_batch_norm = np.zeros_like(img_batch, dtype='float64')
    img_batch_norm[:, :, :, 0] = (img_batch[:, :, :, 0] - img_mean[0]) / img_std[0]
    img_batch_norm[:, :, :, 1] = (img_batch[:, :, :, 1] - img_mean[1]) / img_std[1]
    img_batch_norm[:, :, :, 2] = (img_batch[:, :, :, 2] - img_mean[2]) / img_std[2]
    return img_batch_norm, img_batch

def get_fgbg_cues(cues, H_fg, H_bg, class_inds, indices):
    n_seg_classes = H_fg.shape[1] + 1
    localization_onehot = np.zeros((H_fg.shape[0], n_seg_classes, H_fg.shape[2], H_fg.shape[3]), dtype='int64')
    localization = np.zeros_like(localization_onehot)
    # Obtain localization cues
    # - Background
    for iter_input_image in range(H_bg.shape[0]):
        # grad = scipy.ndimage.median_filter(H_bg[iter_input_image], 3)
        grad = scipy.ndimage.median_filter(np.sum(H_bg[iter_input_image], axis=0), 3)
        thr = np.sort(grad.ravel())[int(0.1 * grad.shape[0] * grad.shape[1])]
        localization[iter_input_image, 0] = grad < thr
    # - Foreground
    for i in range(1, n_seg_classes):
        localization[:, i] = H_fg[:, i-1] > 0.2 * np.max(H_fg[:, i-1])

    # Solve overlap conflicts
    class_rank = np.argsort(-np.sum(np.sum(localization, axis=-1), axis=-1))  # from largest to smallest masks
    localization_ind = np.zeros((H_fg.shape[0], H_fg.shape[2], H_fg.shape[3]), dtype='int64')
    img_inds = np.arange(class_rank.shape[0])
    for iter_class in range(class_rank.shape[1]):
        cur_masks = localization[img_inds, class_rank[:, iter_class]]
        localization_ind *= np.int64(cur_masks == 0)
        localization_ind += np.expand_dims(np.expand_dims(class_rank[:, iter_class]+1, axis=1), axis=2) * cur_masks
    for iter_class in range(class_rank.shape[1]):
        localization_onehot[:, iter_class] = localization_ind == (iter_class+1)
    # Save true one-hot encoded values
    for i,x in enumerate(indices):
        cues['%d_labels' % x] = class_inds[i]
        cues['%d_cues' % x] = np.array(np.where(localization_onehot[i])) # class is front
    return cues

def get_fg_cues(cues, H_fg, class_inds, indices):
    n_seg_classes = H_fg.shape[1]
    localization_onehot = np.zeros((H_fg.shape[0], n_seg_classes, H_fg.shape[2], H_fg.shape[3]), dtype='int64')
    localization = np.zeros_like(localization_onehot)
    # Obtain localization cues
    for i in range(n_seg_classes):
        localization[:, i] = H_fg[:, i] > 0.2 * np.max(H_fg[:, i])

    # Solve overlap conflicts
    class_rank = np.argsort(-np.sum(np.sum(localization, axis=-1), axis=-1))  # from largest to smallest masks
    localization_ind = np.zeros((H_fg.shape[0], H_fg.shape[2], H_fg.shape[3]), dtype='int64')
    img_inds = np.arange(class_rank.shape[0])
    for iter_class in range(class_rank.shape[1]):
        cur_masks = localization[img_inds, class_rank[:, iter_class]]
        localization_ind *= np.int64(cur_masks == 0)
        localization_ind += np.expand_dims(np.expand_dims(class_rank[:, iter_class]+1, axis=1), axis=2) * cur_masks
    for iter_class in range(class_rank.shape[1]):
        localization_onehot[:, iter_class] = localization_ind == (iter_class+1)
    # Save true one-hot encoded values
    for i,x in enumerate(indices):
        cues['%d_labels' % x] = class_inds[i]
        cues['%d_cues' % x] = np.array(np.where(localization_onehot[i])) # class is front
    return cues

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