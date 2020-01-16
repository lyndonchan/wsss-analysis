import time
import skimage.io as imgio
import pandas as pd
import numpy.matlib

from adp_cues import ADPCues
from utilities import *
from dataset import Dataset

MODEL_CNN_ROOT = '../database/models_cnn'
MODEL_WSSS_ROOT = '../database/models_wsss'

def segment(dataset, model_type, batch_size, set_name=None, should_saveimg=True, is_verbose=True):
    assert(dataset in ['ADP', 'VOC2012', 'DeepGlobe_train75', 'DeepGlobe_train37.5'])
    assert(model_type in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'X1.7', 'M7', 'M7bg', 'VGG16', 'VGG16bg'])
    assert(os.path.exists(os.path.exists(os.path.join(MODEL_CNN_ROOT, dataset + '_' + model_type))))
    assert(batch_size > 0)
    assert(set_name in [None, 'tuning', 'segtest'])
    assert(type(is_verbose) is bool)
    if model_type in ['VGG16', 'VGG16bg']:
        img_size = 321
    else:
        img_size = 224
    sess_id = dataset + '_' + model_type
    model_dir = os.path.join(MODEL_CNN_ROOT, sess_id)

    if is_verbose:
        print('Predict: dataset=' + dataset + ', model=' + model_type)

    database_dir = os.path.join(os.path.dirname(os.getcwd()), 'database')
    if dataset == 'ADP':
        segment_adp(sess_id, model_type, batch_size, img_size, set_name, should_saveimg, is_verbose)
        return
    elif dataset == 'VOC2012':
        devkit_dir = os.path.join(database_dir, 'VOCdevkit', 'VOC2012')
        fgbg_modes = ['fg', 'bg']
        OVERLAY_R = 0.75
    elif 'DeepGlobe' in dataset:
        devkit_dir = os.path.join(database_dir, 'DGdevkit')
        fgbg_modes = ['fg']
        OVERLAY_R = 0.25
    img_dir = os.path.join(devkit_dir, 'JPEGImages')
    gt_dir = os.path.join(devkit_dir, 'SegmentationClassAug')

    out_dir = os.path.join('./out', sess_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    eval_dir = os.path.join('./eval', sess_id)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Load network and thresholds
    mdl = {}
    thresholds = {}
    alpha = {}
    final_layer = {}
    for fgbg_mode in fgbg_modes:
        mdl[fgbg_mode] = build_model(model_dir, sess_id)
        thresholds[fgbg_mode] = load_thresholds(model_dir, sess_id)
        thresholds[fgbg_mode] = np.maximum(np.minimum(thresholds[fgbg_mode], 0), 1 / 3)
        alpha[fgbg_mode], final_layer[fgbg_mode] = get_grad_cam_weights(mdl[fgbg_mode],
                                                                        np.zeros((1, img_size, img_size, 3)))

    # Load data and classes
    ds = Dataset(data_type=dataset, size=img_size, batch_size=batch_size)
    class_names, seg_class_names = load_classes(dataset)
    colours = load_colours(dataset)
    if 'DeepGlobe' in dataset:
        colours = colours[:-1]
    gen_curr = ds.set_gens[ds.sets[ds.is_evals.index(True)]]

    # Process images in batches
    intersects = np.zeros((len(colours)))
    unions = np.zeros((len(colours)))
    confusion_matrix = np.zeros((len(colours), len(colours)))
    gt_count = np.zeros((len(colours)))
    n_batches = len(gen_curr.filenames) // batch_size + 1
    for iter_batch in range(n_batches):
        batch_start_time = time.time()
        print('Batch #%d of %d' % (iter_batch + 1, n_batches))
        start_idx = iter_batch * batch_size
        end_idx = min(start_idx + batch_size - 1, len(gen_curr.filenames) - 1)
        cur_batch_sz = end_idx - start_idx + 1

        # Image reading
        start_time = time.time()
        img_batch_norm, img_batch = read_batch(gen_curr.directory, gen_curr.filenames[start_idx:end_idx + 1],
                                               cur_batch_sz, (img_size, img_size), dataset)
        print('\tImage read time: %0.5f seconds (%0.5f seconds / image)' % (time.time() - start_time,
                                                                          (time.time() - start_time) / cur_batch_sz))

        # Generate patch confidence scores
        start_time = time.time()
        predicted_scores = {}
        is_pass_threshold = {}
        for fgbg_mode in fgbg_modes:
            predicted_scores[fgbg_mode] = mdl[fgbg_mode].predict(img_batch_norm)
            is_pass_threshold[fgbg_mode] = np.greater_equal(predicted_scores[fgbg_mode], thresholds[fgbg_mode])
        print('\tGenerating patch confidence scores time: %0.5f seconds (%0.5f seconds / image)' % (
        time.time() - start_time,
        (time.time() - start_time) / cur_batch_sz))

        # Generate Grad-CAM
        start_time = time.time()
        H = {}
        for fgbg_mode in fgbg_modes:
            H[fgbg_mode] = grad_cam(mdl[fgbg_mode], alpha[fgbg_mode], img_batch_norm, is_pass_threshold[fgbg_mode],
                                    final_layer[fgbg_mode], predicted_scores[fgbg_mode], orig_sz=[img_size, img_size],
                                    should_upsample=True)
            H[fgbg_mode] = np.transpose(H[fgbg_mode], (0, 3, 1, 2))
        print('\tGenerating Grad-CAM time: %0.5f seconds (%0.5f seconds / image)' % (time.time() - start_time,
                                                                    (time.time() - start_time) / cur_batch_sz))

        # Modify fg Grad-CAM with bg activation
        start_time = time.time()
        if dataset == 'VOC2012':
            Y_gradcam = np.zeros((cur_batch_sz, len(seg_class_names), img_size, img_size))
            mode = 'mult'
            if mode == 'mult':
                X_bg = np.sum(H['bg'], axis=1)
                Y_gradcam[:, 0] = 0.15 * scipy.special.expit(np.max(X_bg) - X_bg)
            Y_gradcam[:, 1:] = H['fg']
        elif 'DeepGlobe' in dataset:
            Y_gradcam = H['fg'][:, :-1, :, :]
        print('\tFg/Bg modifications time: %0.5f seconds (%0.5f seconds / image)' % (time.time() - start_time,
                                                                         (time.time() - start_time) / cur_batch_sz))

        # FC-CRF
        start_time = time.time()
        if dataset == 'VOC2012':
            dcrf_config = np.array([3 / 4, 3, 80 / 4, 13, 10, 10])  # test (since 2448 / 500 = 4.896 ~= 4)
        elif 'DeepGlobe' in dataset:
            dcrf_config = np.array([3, 3, 80, 13, 10, 10])  # test
        Y_crf = dcrf_process(Y_gradcam, img_batch, dcrf_config)
        print('\tCRF time: %0.5f seconds (%0.5f seconds / image)' % (time.time() - start_time,
                                                                     (time.time() - start_time) / cur_batch_sz))
        elapsed_time = time.time() - batch_start_time
        print('\tElapsed time: %0.5f seconds (%0.5f seconds / image)' % (elapsed_time, elapsed_time / cur_batch_sz))

        if dataset == 'VOC2012':
            for iter_file, filename in enumerate(gen_curr.filenames[start_idx:end_idx + 1]):
                gt_filepath = os.path.join(gt_dir, filename.replace('.jpg', '.png'))
                gt_idx = cv2.cvtColor(cv2.imread(gt_filepath), cv2.COLOR_RGB2BGR)[:, :, 0]
                pred_idx = cv2.resize(np.uint8(Y_crf[iter_file]), (gt_idx.shape[1], gt_idx.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                pred_segmask = np.zeros((gt_idx.shape[0], gt_idx.shape[1], 3))
                for k in range(len(colours)):
                    intersects[k] += np.sum((gt_idx == k) & (pred_idx == k))
                    unions[k] += np.sum((gt_idx == k) | (pred_idx == k))
                    confusion_matrix[k, :] += np.bincount(pred_idx[gt_idx == k], minlength=len(colours))
                    pred_segmask += np.expand_dims(pred_idx == k, axis=2) * \
                                    np.expand_dims(np.expand_dims(colours[k], axis=0), axis=0)
                    gt_count[k] += np.sum(gt_idx == k)
                if should_saveimg:
                    orig_filepath = os.path.join(img_dir, filename)
                    orig_img = cv2.cvtColor(cv2.imread(orig_filepath), cv2.COLOR_RGB2BGR)
                    imgio.imsave(os.path.join(out_dir, filename.replace('.jpg', '') + '.png'), pred_segmask / 256.0)
                    imgio.imsave(os.path.join(out_dir, filename.replace('.jpg', '') + '_overlay.png'),
                                 (1 - OVERLAY_R) * orig_img / 256.0 +
                                 OVERLAY_R * pred_segmask / 256.0)
        elif 'DeepGlobe' in dataset:
            for iter_file, filename in enumerate(gen_curr.filenames[start_idx:end_idx + 1]):
                gt_filepath = os.path.join(gt_dir, filename.replace('.jpg', '.png'))
                gt_curr = cv2.cvtColor(cv2.imread(gt_filepath), cv2.COLOR_RGB2BGR)
                gt_r = gt_curr[:, :, 0]
                gt_g = gt_curr[:, :, 1]
                gt_b = gt_curr[:, :, 2]
                pred_idx = cv2.resize(np.uint8(Y_crf[iter_file]), (gt_curr.shape[1], gt_curr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                pred_segmask = np.zeros((gt_curr.shape[0], gt_curr.shape[1], 3))
                for k, gt_colour in enumerate(colours):
                    gt_mask = (gt_r == gt_colour[0]) & (gt_g == gt_colour[1]) & (gt_b == gt_colour[2])
                    pred_mask = pred_idx == k
                    intersects[k] += np.sum(gt_mask & pred_mask)
                    unions[k] += np.sum(gt_mask | pred_mask)
                    confusion_matrix[k, :] += np.bincount(pred_idx[gt_mask], minlength=len(colours))
                    pred_segmask += np.expand_dims(pred_mask, axis=2) * \
                                    np.expand_dims(np.expand_dims(colours[k], axis=0), axis=0)
                    gt_count[k] += np.sum(gt_mask)
                if should_saveimg:
                    orig_filepath = os.path.join(img_dir, filename)
                    orig_img = cv2.cvtColor(cv2.imread(orig_filepath), cv2.COLOR_RGB2BGR)
                    orig_img = cv2.resize(orig_img, (orig_img.shape[0] // 4, orig_img.shape[1] // 4))
                    pred_segmask = cv2.resize(pred_segmask, (pred_segmask.shape[0] // 4, pred_segmask.shape[1] // 4),
                                              interpolation=cv2.INTER_NEAREST)
                    imgio.imsave(os.path.join(out_dir, filename.replace('.jpg', '') + '.png'), pred_segmask / 256.0)
                    imgio.imsave(os.path.join(out_dir, filename.replace('.jpg', '') + '_overlay.png'),
                                 (1 - OVERLAY_R) * orig_img / 256.0 + OVERLAY_R * pred_segmask / 256.0)
    mIoU = np.mean(intersects / (unions + 1e-7))
    df = pd.DataFrame({'Class': seg_class_names + ['Mean'], 'IoU': list(intersects / (unions + 1e-7)) + [mIoU]},
                      columns=['Class', 'IoU'])
    xlsx_path = os.path.join(eval_dir, 'metrics_' + sess_id + '.xlsx')
    df.to_excel(xlsx_path)

    count_mat = np.transpose(np.matlib.repmat(gt_count, len(colours), 1))
    title = "Confusion matrix\n"
    xlabel = 'Prediction'  # "Labels"
    ylabel = 'Ground-Truth'  # "Labels"
    xticklabels = seg_class_names
    yticklabels = seg_class_names
    heatmap(confusion_matrix / (count_mat + 1e-7), title, xlabel, ylabel, xticklabels, yticklabels,
            rot_angle=45)
    plt.savefig(os.path.join(eval_dir, 'confusion_' + sess_id + '.png'), dpi=96,
                format='png', bbox_inches='tight')

    title = "Confusion matrix\n"
    xlabel = 'Prediction'  # "Labels"
    ylabel = 'Ground-Truth'  # "Labels"
    if dataset == 'VOC2012':
        xticklabels = seg_class_names[1:]
        yticklabels = seg_class_names[1:]
        heatmap(confusion_matrix[1:, 1:] / (count_mat[1:, 1:] + 1e-7), title, xlabel, ylabel, xticklabels, yticklabels,
                rot_angle=45)
    elif 'DeepGlobe' in dataset:
        xticklabels = seg_class_names[:-2]
        yticklabels = seg_class_names[:-1]
        heatmap(confusion_matrix[:-1, :-1] / (count_mat[:-1, :-1] + 1e-7), title, xlabel, ylabel, xticklabels,
                yticklabels,
                rot_angle=45)
    plt.savefig(os.path.join(eval_dir, 'confusion_fore_' + sess_id + '.png'), dpi=96,
                format='png', bbox_inches='tight')
    plt.close()

def segment_adp(sess_id, model_type, batch_size, size, set_name, should_saveimg, is_verbose):
    ac = ADPCues(sess_id, batch_size, size, model_dir=MODEL_CNN_ROOT)
    OVERLAY_R = 0.75

    # Load network and thresholds
    ac.build_model()

    # Load images
    if is_verbose:
        print('\tGetting Grad-CAM weights for given network')
    alpha = ac.get_grad_cam_weights(np.zeros((1, size, size, 3)))

    # Read in image names
    img_names = ac.get_img_names(set_name)

    # Process images in batches
    confusion_matrix = {}
    gt_count = {}
    out_dirs = {}
    for htt_class in ['morph', 'func']:
        confusion_matrix[htt_class] = np.zeros((len(ac.classes['valid_' + htt_class]), len(ac.classes['valid_' + htt_class])))
        gt_count[htt_class] = np.zeros((len(ac.classes['valid_' + htt_class])))
        out_dirs[htt_class] = os.path.join('out', 'ADP-' + htt_class + '_' + set_name + '_' + model_type)
        if not os.path.exists(out_dirs[htt_class]):
            os.makedirs(out_dirs[htt_class])
    n_batches = len(img_names) // batch_size + 1
    for iter_batch in range(n_batches):
        batch_start_time = time.time()
        print('\tBatch #%d of %d' % (iter_batch + 1, n_batches))
        start_idx = iter_batch * batch_size
        end_idx = min(start_idx + batch_size - 1, len(img_names) - 1)
        cur_batch_sz = end_idx - start_idx + 1

        # Image reading
        start_time = time.time()
        img_batch_norm, img_batch = ac.read_batch(img_names[start_idx:end_idx + 1])
        print('\t\tImage read time: %0.5f seconds (%0.5f seconds / image)' % (
        time.time() - start_time, (time.time() - start_time) / cur_batch_sz))

        # Generate patch confidence scores
        start_time = time.time()
        predicted_scores = ac.model.predict(img_batch_norm)
        is_pass_threshold = np.greater_equal(predicted_scores, ac.thresholds)
        print('\t\tGenerating patch confidence scores time: %0.5f seconds (%0.5f seconds / image)' % (
        time.time() - start_time, (time.time() - start_time) / cur_batch_sz))

        # Generate Grad-CAM
        start_time = time.time()
        H = grad_cam(ac.model, alpha, img_batch_norm, is_pass_threshold, ac.final_layer, predicted_scores,
                     orig_sz=[size, size], should_upsample=True)
        H = np.transpose(H, (0, 3, 1, 2))
        # Split Grad-CAM into {morph, func}
        H_split = {}
        H_split['morph'], H_split['func'] = split_by_httclass(H, ac.classes['all'], ac.classes['morph'], ac.classes['func'])
        is_pass = {}
        is_pass['morph'], is_pass['func'] = split_by_httclass(is_pass_threshold, ac.classes['all'], ac.classes['morph'], ac.classes['func'])
        print('\t\tGenerating Grad-CAM time: %0.5f seconds (%0.5f seconds / image)' % (
        time.time() - start_time, (time.time() - start_time) / cur_batch_sz))

        # Modify Grad-CAM for each HTT type separately
        Y_gradcam = {}
        Y_csgc = {}
        Y_crf = {}
        for htt_class in ['morph', 'func']:
            Y_gradcam[htt_class] = np.zeros((cur_batch_sz, len(ac.classes['valid_' + htt_class]), size, size))
            Y_gradcam[htt_class][:, ac.classinds[htt_class + '2valid']] = H[:, ac.classinds['all2' + htt_class]]

            # Inter-HTT Adjustments
            start_time = time.time()
            if htt_class == 'morph':
                Y_gradcam[htt_class] = modify_by_htt(Y_gradcam[htt_class], img_batch, ac.classes['valid_' + htt_class])
            elif htt_class == 'func':
                adipose_inds = [i for i, x in enumerate(ac.classes['morph']) if x in ['A.W', 'A.B', 'A.M']]
                gradcam_adipose = Y_gradcam['morph'][:, adipose_inds]
                Y_gradcam[htt_class] = modify_by_htt(Y_gradcam[htt_class], img_batch, ac.classes['valid_' + htt_class],
                                                     gradcam_adipose=gradcam_adipose)
            Y_csgc[htt_class] = get_cs_gradcam(Y_gradcam[htt_class], ac.classes['valid_' + htt_class], htt_class)
            print('\t\t\tInter-HTT adjustments time [%s]: %0.5f seconds (%0.5f seconds / image)' % (htt_class,
            time.time() - start_time, (time.time() - start_time) / cur_batch_sz))

            # FC-CRF
            start_time = time.time()
            dcrf_config = np.load(os.path.join(MODEL_WSSS_ROOT, htt_class + '_optimal_pcc.npy'))[0]
            Y_crf[htt_class] = dcrf_process(Y_csgc[htt_class], img_batch, dcrf_config)
            print('\t\t\tCRF time [%s]: %0.5f seconds (%0.5f seconds / image)' % (htt_class,
            time.time() - start_time, (time.time() - start_time) / cur_batch_sz))

            # Update evaluation performance
            _, gt_batch = read_batch(os.path.join(ac.gt_root, 'ADP-' + htt_class), img_names[start_idx:end_idx + 1], cur_batch_sz,
                                     [1088, 1088], 'ADP')
            for iter_img in range(cur_batch_sz):
                pred_idx = cv2.resize(Y_crf[htt_class][iter_img], dsize=(1088, 1088), interpolation=cv2.INTER_NEAREST)
                gt_r = gt_batch[iter_img][:, :, 0]
                gt_g = gt_batch[iter_img][:, :, 1]
                gt_b = gt_batch[iter_img][:, :, 2]
                pred_segmask = np.zeros((1088, 1088, 3))
                for k, gt_colour in enumerate(ac.colours[htt_class]):
                    gt_mask = (gt_r == gt_colour[0]) & (gt_g == gt_colour[1]) & (gt_b == gt_colour[2])
                    pred_mask = pred_idx == k
                    confusion_matrix[htt_class][k, :] += np.bincount(pred_idx[gt_mask],
                                                                     minlength=len(ac.classes['valid_' + htt_class]))
                    ac.intersects[htt_class][k] += np.sum(gt_mask & pred_mask)
                    ac.unions[htt_class][k] += np.sum(gt_mask | pred_mask)
                    pred_segmask += np.expand_dims(pred_mask, axis=2) * \
                                    np.expand_dims(np.expand_dims(ac.colours[htt_class][k], axis=0), axis=0)
                    gt_count[htt_class][k] += np.sum(gt_mask)
                if should_saveimg:
                    orig_filepath = os.path.join(ac.img_dir, img_names[start_idx + iter_img])
                    orig_img = cv2.cvtColor(cv2.imread(orig_filepath), cv2.COLOR_RGB2BGR)
                    pred_segmask_small = cv2.resize(pred_segmask, (orig_img.shape[0], orig_img.shape[1]),
                                                    interpolation=cv2.INTER_NEAREST)
                    imgio.imsave(
                        os.path.join(out_dirs[htt_class], img_names[start_idx + iter_img].replace('.png', '') + '.png'),
                        pred_segmask_small / 256.0)
                    imgio.imsave(os.path.join(out_dirs[htt_class],
                                              img_names[start_idx + iter_img].replace('.png', '') + '_overlay.png'),
                                 (1 - OVERLAY_R) * orig_img / 256.0 +
                                 OVERLAY_R * pred_segmask_small / 256.0)
        elapsed_time = time.time() - batch_start_time
        print('\tElapsed time: %0.5f seconds (%0.5f seconds / image)' % (elapsed_time, elapsed_time / cur_batch_sz))

    for htt_class in ['morph', 'func']:
        mIoU = np.mean(ac.intersects[htt_class] / (ac.unions[htt_class] + 1e-7))
        df = pd.DataFrame({'Class': ac.classes['valid_' + htt_class] + ['Mean'],
                           'IoU': list(ac.intersects[htt_class] / (ac.unions[htt_class] + 1e-7)) + [mIoU]},
                          columns=['Class', 'IoU'])
        eval_dir = os.path.join('eval', 'ADP-' + htt_class + '_' + set_name + '_' + model_type)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        xlsx_path = os.path.join(eval_dir, 'metrics_ADP-' + htt_class + '_' + model_type + '.xlsx')
        df.to_excel(xlsx_path)

        count_mat = np.transpose(np.matlib.repmat(gt_count[htt_class], len(ac.classes['valid_' + htt_class]), 1))
        title = "Confusion matrix\n"
        xlabel = 'Prediction'  # "Labels"
        ylabel = 'Ground-Truth'  # "Labels"
        xticklabels = ac.classes['valid_' + htt_class]
        yticklabels = ac.classes['valid_' + htt_class]
        heatmap(confusion_matrix[htt_class] / (count_mat + 1e-7), title, xlabel, ylabel, xticklabels, yticklabels,
                rot_angle=45)
        plt.savefig(os.path.join(eval_dir, 'confusion_ADP-' + htt_class + '_' + model_type + '.png'), dpi=96,
                    format='png', bbox_inches='tight')
        plt.close()

        title = "Confusion matrix\n"
        xlabel = 'Prediction'  # "Labels"
        ylabel = 'Ground-Truth'  # "Labels"
        if htt_class == 'morph':
            xticklabels = ac.classes['valid_' + htt_class][1:]
            yticklabels = ac.classes['valid_' + htt_class][1:]
            heatmap(confusion_matrix[htt_class][1:, 1:] / (count_mat[1:, 1:] + 1e-7), title, xlabel, ylabel,
                    xticklabels, yticklabels, rot_angle=45)
        elif htt_class == 'func':
            xticklabels = ac.classes['valid_' + htt_class][2:]
            yticklabels = ac.classes['valid_' + htt_class][2:]
            heatmap(confusion_matrix[htt_class][2:, 2:] / (count_mat[2:, 2:] + 1e-7), title, xlabel, ylabel,
                    xticklabels, yticklabels, rot_angle=45)
        plt.savefig(os.path.join(eval_dir, 'confusion_fore_ADP-' + htt_class + '_' + model_type + '.png'), dpi=96,
                    format='png', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # ADP
    # segment(dataset='ADP', model_type='VGG16', batch_size=16, set_name='tuning', should_saveimg=True, is_verbose=True)
    # segment(dataset='ADP', model_type='VGG16', batch_size=16, set_name='segtest', should_saveimg=True, is_verbose=True)
    # segment(dataset='ADP', model_type='X1.7', batch_size=16, set_name='tuning', should_saveimg=True, is_verbose=True)
    # segment(dataset='ADP', model_type='X1.7', batch_size=16, set_name='segtest', should_saveimg=True, is_verbose=True)

    # VOC2012
    # segment(dataset='VOC2012', model_type='VGG16', batch_size=16, should_saveimg=True, is_verbose=True)
    # segment(dataset='VOC2012', model_type='M7', batch_size=16, should_saveimg=True, is_verbose=True)

    # DeepGlobe
    # segment(dataset='DeepGlobe_train75', model_type='VGG16', batch_size=16, should_saveimg=True, is_verbose=True)
    # segment(dataset='DeepGlobe_train75', model_type='M7', batch_size=16, should_saveimg=True, is_verbose=True)
    # segment(dataset='DeepGlobe_train37.5', model_type='VGG16', batch_size=16, should_saveimg=True, is_verbose=True)
    segment(dataset='DeepGlobe_train37.5', model_type='M7', batch_size=16, should_saveimg=True, is_verbose=True)