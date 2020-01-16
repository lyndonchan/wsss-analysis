import pickle
import scipy.io as sio
import skimage.io as imgio
from keras import optimizers
from keras.models import model_from_json
import pandas as pd
import time
import math

from utilities import *
from dataset import Dataset
from adp_cues import ADPCues

LOG_ROOT = './log'
MODEL_ROOT = '../database/models_cnn'
CKPT_ROOT = './ckpt'
OUT_ROOT = './out'
TRAIN_CUES_ROOT = './cues_train'
EVAL_CUES_ROOT = './cues_eval'

def gen_cues(dataset, model_type, batch_size, set_name=None, run_train=True, is_verbose=True):
    assert(dataset in ['ADP', 'VOC2012', 'DeepGlobe_train75', 'DeepGlobe_train37.5'])
    assert(model_type in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'X1.7', 'M7', 'M7bg', 'VGG16', 'VGG16bg'])
    assert(os.path.exists(os.path.join(MODEL_ROOT, dataset + '_' + model_type)))
    assert(batch_size > 0)
    assert(set_name in [None, 'tuning', 'segtest'])
    assert(type(run_train) is bool)
    assert(type(is_verbose) is bool)
    if model_type in ['VGG16', 'VGG16bg']:
        img_size = 321
    else:
        img_size = 224
    seed_size = 41

    if set_name is None:
        sess_id = dataset + '_' + model_type
    else:
        sess_id = dataset + '_' + set_name + '_' + model_type
    model_dir = os.path.join(MODEL_ROOT, dataset + '_' + model_type)
    train_cues_dir = os.path.join(TRAIN_CUES_ROOT, sess_id)
    eval_cues_dir = os.path.join(EVAL_CUES_ROOT, sess_id)
    if run_train:
        file_list = [train_cues_dir]
    else:
        file_list = [eval_cues_dir]
    makedir_if_nexist(file_list)
    if is_verbose:
        if set_name is None:
            print('Evaluate cues: dataset=' + dataset + ', model=' + model_type)
        else:
            print('Evaluate cues: dataset=' + dataset + ', set=' + set_name + ', model=' + model_type)

    # Load data and classes
    if is_verbose:
        print('\tLoading data')
    if dataset == 'ADP':
        if run_train:
            gen_cues_adp(dataset, model_type, sess_id, batch_size, img_size, train_cues_dir, set_name, is_verbose)
        else:
            gen_cues_adp(dataset, model_type, sess_id, batch_size, img_size, eval_cues_dir, set_name, is_verbose)
        return
    ds = Dataset(data_type=dataset, size=img_size, batch_size=batch_size)
    if run_train:
        gen_curr = ds.set_gens[ds.sets[ds.is_evals.index(False)]]
    else:
        gen_curr = ds.set_gens[ds.sets[ds.is_evals.index(True)]]
    img_names = gen_curr.filenames

    # Load and compile models
    def load_model(model_dir, sess_id):
        # Load model
        arch_path = os.path.join(model_dir, sess_id + '.json')
        with open(arch_path, 'r') as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        weights_path = os.path.join(model_dir, sess_id + '.h5')
        model.load_weights(weights_path)
        # Compile model
        opt = optimizers.SGD(lr=0.0, momentum=0.0, decay=0.0, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
        # Get alpha
        final_layer = find_final_layer(model)
        alpha = get_grad_cam_weights(model, final_layer, np.zeros((1, img_size, img_size, 3)))
        # Get thresholds
        thresh_path = os.path.join(model_dir, sess_id + '.mat')
        assert (os.path.exists(thresh_path))
        tmp = sio.loadmat(thresh_path)
        thresholds = tmp.get('optimalScoreThresh')
        return model, alpha, final_layer, thresholds

    # Load fg/bg models
    model = {}
    alpha = {}
    final_layer = {}
    thresholds = {}
    fgbg_dir = {}
    fgbg_sess = {}
    H = {}
    localization_cues = {}
    is_pass_threshold = {}
    if dataset == 'VOC2012':
        fgbg_modes = ['fg', 'bg']
    elif 'DeepGlobe' in dataset:
        fgbg_modes = ['fg']
    for fgbg_mode in fgbg_modes:
        if fgbg_mode == 'fg':
            fgbg_dir[fgbg_mode] = model_dir
            fgbg_sess[fgbg_mode] = sess_id
        elif fgbg_mode == 'bg':
            if 'fg' in sess_id:
                fgbg_dir[fgbg_mode] = model_dir.replace('fg', 'bg')
                fgbg_sess[fgbg_mode] = sess_id.replace('fg', 'bg')
            else:
                fgbg_dir[fgbg_mode] = model_dir.replace('fg', '') + 'bg'
                fgbg_sess[fgbg_mode] = sess_id.replace('fg', '') + 'bg'
        model[fgbg_mode], alpha[fgbg_mode], final_layer[fgbg_mode], thresholds[fgbg_mode] = \
            load_model(fgbg_dir[fgbg_mode], fgbg_sess[fgbg_mode])
    n_batches = math.ceil(len(img_names) / batch_size)
    for iter_batch in range(n_batches):
        start_time = time.time()
        if is_verbose:
            print('\tBatch #%d of %d' % (iter_batch + 1, n_batches))
        start_idx = iter_batch * batch_size
        end_idx = min(start_idx + batch_size - 1, len(img_names) - 1)
        cur_batch_sz = end_idx - start_idx + 1
        if dataset == 'VOC2012':
            dataset_mean = [104, 117, 123]
            dataset_std = [255, 255, 255]
            ignore_ind = None
        elif 'DeepGlobe' in dataset:
            dataset_mean = [0, 0, 0]
            dataset_std = [255, 255, 255]
            ignore_ind = 6
        img_batch_norm, img_batch = read_batch(gen_curr.directory, img_names[start_idx:end_idx + 1], cur_batch_sz,
                                               (img_size, img_size), img_mean=dataset_mean, img_std=dataset_std)

        # Determine passing classes
        for fgbg_mode in fgbg_modes:
            pred_scores = model[fgbg_mode].predict(img_batch_norm)
            keep_inds = np.arange(len(ds.class_names))
            if ignore_ind is not None:
                keep_inds = np.delete(keep_inds, ignore_ind)
                pred_scores = pred_scores[:, keep_inds]
                thresholds[fgbg_mode] = thresholds[fgbg_mode][:, keep_inds]
            is_pass_threshold[fgbg_mode] = np.greater_equal(pred_scores, thresholds[fgbg_mode]) * \
                                           gen_curr.data[start_idx:end_idx+1, keep_inds]

            # Generate CAM/Grad-CAM
            if fgbg_mode == 'fg':
                fg_start_time = time.time()
                mode = 'Grad-CAM'
                if mode == 'Grad-CAM':
                    H[fgbg_mode] = grad_cam(model[fgbg_mode], alpha[fgbg_mode], img_batch_norm, is_pass_threshold[fgbg_mode],
                                            final_layer[fgbg_mode], keep_inds, [img_size, img_size])
                elapsed_time = time.time() - fg_start_time
                if is_verbose:
                    print('\t\tElapsed time (fg): %s seconds (%s seconds/image)' % (elapsed_time, elapsed_time / cur_batch_sz))
            elif fgbg_mode == 'bg':
                bg_start_time = time.time()
                mode = 'Grad-CAM'
                if mode == 'Grad-CAM':
                    H[fgbg_mode] = grad_cam(model[fgbg_mode], alpha[fgbg_mode], img_batch_norm, is_pass_threshold[fgbg_mode],
                                            final_layer[fgbg_mode], keep_inds, [img_size, img_size])
                elapsed_time = time.time() - bg_start_time
                if is_verbose:
                    print('\t\tElapsed time (bg): %s seconds (%s seconds/image)' % (elapsed_time, elapsed_time / cur_batch_sz))
            H[fgbg_mode] = np.transpose(H[fgbg_mode], (0, 3, 1, 2))
            H[fgbg_mode] = resize_stack(H[fgbg_mode], (seed_size, seed_size))

        # Generate localization cues
        if dataset == 'VOC2012':
            class_inds = [np.where(is_pass_threshold['fg'][i])[0] + 1 for i in range(is_pass_threshold['fg'].shape[0])]
        elif 'DeepGlobe' in dataset:
            class_inds = [np.where(is_pass_threshold['fg'][i])[0] for i in range(is_pass_threshold['fg'].shape[0])]
        list_idx = list(range(start_idx, end_idx+1))
        if dataset == 'VOC2012':
            localization_cues = get_fgbg_cues(localization_cues, H['fg'], H['bg'], class_inds, list_idx)
        elif 'DeepGlobe' in dataset:
            localization_cues = get_fg_cues(localization_cues, H['fg'], class_inds, list_idx)
        elapsed_time = time.time() - start_time
        if is_verbose:
            print('\t\tElapsed time: %s seconds (%s seconds/image)' % (elapsed_time, elapsed_time / cur_batch_sz))
    print('Saving localization cues')
    if run_train:
        pickle.dump(localization_cues, open(os.path.join(train_cues_dir, 'localization_cues.pickle'), 'wb'))
    else:
        pickle.dump(localization_cues, open(os.path.join(eval_cues_dir, 'localization_cues_val.pickle'), 'wb'))

def gen_cues_adp(dataset, model_type, sess_id, batch_size, size, cues_dir, set_name, is_verbose):
    ac = ADPCues(dataset + '_' + model_type, batch_size, size, model_dir=MODEL_ROOT)
    seed_size = 41

    # Load network and thresholds
    out_dirs = {}
    out_dirs['morph'] = os.path.join(cues_dir, 'morph')
    if not os.path.exists(out_dirs['morph']):
        os.makedirs(out_dirs['morph'])
    out_dirs['func'] = os.path.join(cues_dir, 'func')
    if not os.path.exists(out_dirs['func']):
        os.makedirs(out_dirs['func'])
    ac.build_model()

    # Load images
    if not os.path.exists('data'):
        os.makedirs('data')
    if is_verbose:
        print('\tGetting Grad-CAM weights for given network')
    alpha = ac.get_grad_cam_weights(np.zeros((1, size, size, 3)))

    # Read in image names
    img_names = ac.get_img_names(set_name)

    # Process images in batches
    n_batches = len(img_names) // batch_size + 1
    for iter_batch in range(n_batches):
        start_time = time.time()
        if is_verbose:
            print('\tBatch #%d of %d' % (iter_batch + 1, n_batches))
        start_idx = iter_batch * batch_size
        end_idx = min(start_idx + batch_size - 1, len(img_names) - 1)
        cur_batch_sz = end_idx - start_idx + 1
        img_batch_norm, img_batch = ac.read_batch(img_names[start_idx:end_idx + 1])

        # Determine passing classes
        predicted_scores = ac.model.predict(img_batch_norm)
        is_pass_threshold = np.greater_equal(predicted_scores, ac.thresholds)

        # Generate Grad-CAM
        H = ac.grad_cam(alpha, img_batch_norm, is_pass_threshold)
        H = np.transpose(H, (0, 3, 1, 2))
        H = resize_stack(H, (seed_size, seed_size))

        # Split Grad-CAM into {morph, func}
        H_split = {}
        H_split['morph'], H_split['func'] = ac.split_by_httclass(H)
        is_pass = {}
        is_pass['morph'], is_pass['func'] = ac.split_by_httclass(is_pass_threshold)

        # Modify Grad-CAM for each HTT type separately
        seeds = {}
        for htt_class in ['morph', 'func']:
            seeds[htt_class] = np.zeros((cur_batch_sz, len(ac.classes['valid_' + htt_class]), seed_size, seed_size))
            seeds[htt_class][:, ac.classinds[htt_class + '2valid']] = H[:, ac.classinds['all2' + htt_class]]
            class_inds = [ac.classinds_arr[htt_class + '2valid'][is_pass[htt_class][i]] for i in range(cur_batch_sz)]

            # Modify heatmaps
            if htt_class == 'morph':
                seeds[htt_class] = ac.modify_by_htt(seeds[htt_class], img_batch, ac.classes['valid_' + htt_class])
            elif htt_class == 'func':
                class_inds = [np.append(1, x) for x in class_inds]
                adipose_inds = [i for i, x in enumerate(ac.classes['morph']) if x in ['A.W', 'A.B', 'A.M']]
                gradcam_adipose = seeds['morph'][:, adipose_inds]
                seeds[htt_class] = ac.modify_by_htt(seeds[htt_class], img_batch, ac.classes['valid_' + htt_class],
                                                 gradcam_adipose=gradcam_adipose)

            # Save localization cues
            ac.update_cues(seeds[htt_class], class_inds, htt_class, list(range(start_idx, end_idx + 1)))
        elapsed_time = time.time() - start_time
        if is_verbose:
            print('\t\tElapsed time: %s seconds (%s seconds/image)' % (elapsed_time, elapsed_time / cur_batch_sz))
    if is_verbose:
        print('\tSaving localization cues')
    pickle.dump(ac.cues['morph'], open(os.path.join(out_dirs['morph'], 'localization_cues.pickle'), 'wb'))
    pickle.dump(ac.cues['func'], open(os.path.join(out_dirs['func'], 'localization_cues.pickle'), 'wb'))

def eval_cues(dataset, model_type, batch_size, set_name=None, run_train=False, is_verbose=True):
    assert(dataset in ['ADP', 'VOC2012', 'DeepGlobe_train75', 'DeepGlobe_train37.5'])
    assert(model_type in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'X1.7', 'M7', 'M7bg', 'VGG16', 'VGG16bg'])
    assert(os.path.exists(os.path.join(MODEL_ROOT, dataset + '_' + model_type)))
    assert(batch_size > 0)
    assert(set_name in [None, 'tuning', 'segtest'])
    assert(type(run_train) is bool)
    assert(type(is_verbose) is bool)
    if model_type in ['VGG16', 'VGG16bg']:
        img_size = 321
    else:
        img_size = 224
    seed_size = 41

    if dataset == 'VOC2012':
        OVERLAY_R = 0.75
    elif 'DeepGlobe' in dataset:
        OVERLAY_R = 0.25

    if set_name is None:
        sess_id = dataset + '_' + model_type
    else:
        sess_id = dataset + '_' + set_name + '_' + model_type
    eval_cues_dir = os.path.join(EVAL_CUES_ROOT, sess_id)
    out_dir = os.path.join(OUT_ROOT, sess_id)
    makedir_if_nexist([out_dir])
    if is_verbose:
        if set_name is None:
            print('Evaluate cues: dataset=' + dataset + ', model=' + model_type)
        else:
            print('Evaluate cues: dataset=' + dataset + ', set=' + set_name + ', model=' + model_type)
    # Load data and classes
    if is_verbose:
        print('\tLoading data')
    if dataset == 'ADP':
        eval_cues_adp(dataset, model_type, sess_id, batch_size, img_size, set_name, is_verbose)
        return
    makedir_if_nexist([eval_cues_dir])
    ds = Dataset(data_type=dataset, size=img_size, batch_size=batch_size)
    eval_set = ds.sets[ds.is_evals.index(True)]
    gen_eval = ds.set_gens[eval_set]
    colours = get_colours(dataset)

    cues_path = os.path.join(eval_cues_dir, 'localization_cues_val.pickle')

    if not os.path.exists(cues_path):
        gen_cues(dataset, model_type, batch_size, run_train=False, is_verbose=is_verbose)
    localization_cues = pickle.load(open(cues_path, "rb"), encoding="iso-8859-1")

    gt_dir = os.path.join(os.path.dirname(gen_eval.directory), 'SegmentationClassAug')
    if dataset == 'VOC2012':
        seg_class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor']  # 20+1 classes
        ignore_ind = None
    if 'DeepGlobe' in dataset:
        seg_class_names = ds.class_names[:-1]
        colours = colours[:-1]
        ignore_ind = 6
    intersects = np.zeros((len(colours)))
    unions = np.zeros((len(colours)))
    for iter_file, filename in enumerate(gen_eval.filenames):
        if is_verbose:
            print('\tImage #%d of %d' % (iter_file+1, len(gen_eval.filenames)))
        start_time = time.time()
        if dataset == 'VOC2012':
            gt_filepath = os.path.join(gt_dir, filename.replace('.jpg', '.png'))
            gt_idx = cv2.cvtColor(cv2.imread(gt_filepath), cv2.COLOR_RGB2BGR)[:, :, 0]
            cues_i = localization_cues['%s_cues' % iter_file]
            cues_pred = np.zeros([seed_size, seed_size, len(colours)])
            cues_pred[cues_i[1], cues_i[2], cues_i[0]] = 1.0
            cues_pred_max = np.argmax(cues_pred, axis=-1)
            pred_idx = cv2.resize(np.uint8(cues_pred_max), (gt_idx.shape[1], gt_idx.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            pred_segmask = np.zeros((gt_idx.shape[0], gt_idx.shape[1], 3))
            for k in range(len(colours)):
                intersects[k] += np.sum((gt_idx == k) & (pred_idx == k))
                unions[k] += np.sum((gt_idx == k) | (pred_idx == k))
                pred_segmask += np.expand_dims(pred_idx == k, axis=2) * \
                                np.expand_dims(np.expand_dims(colours[k], axis=0), axis=0)
        elif 'DeepGlobe' in dataset:
            gt_filepath = os.path.join(gt_dir, filename.replace('.jpg', '.png'))
            gt_curr = cv2.cvtColor(cv2.imread(gt_filepath), cv2.COLOR_RGB2BGR)
            cues_i = localization_cues['%s_cues' % iter_file]
            cues_pred = np.zeros([seed_size, seed_size, len(colours)])
            cues_pred[cues_i[1], cues_i[2], cues_i[0]] = 1.0
            cues_pred_max = np.argmax(cues_pred, axis=-1)
            cues_pred_max[np.sum(cues_pred, axis=-1) == 0] = ignore_ind
            pred_idx = cv2.resize(np.uint8(cues_pred_max), (gt_curr.shape[1], gt_curr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            pred_segmask = np.zeros((gt_curr.shape[0], gt_curr.shape[1], 3))
            gt_r = gt_curr[:, :, 0]
            gt_g = gt_curr[:, :, 1]
            gt_b = gt_curr[:, :, 2]
            for k, gt_colour in enumerate(colours):
                gt_mask = (gt_r == gt_colour[0]) & (gt_g == gt_colour[1]) & (gt_b == gt_colour[2])
                pred_mask = pred_idx == k
                intersects[k] += np.sum(gt_mask & pred_mask)
                unions[k] += np.sum(gt_mask | pred_mask)
                pred_segmask += np.expand_dims(pred_idx == k, axis=2) * \
                                np.expand_dims(np.expand_dims(colours[k], axis=0), axis=0)
        orig_filepath = os.path.join(gen_eval.directory, filename)
        orig_img = cv2.cvtColor(cv2.imread(orig_filepath), cv2.COLOR_RGB2BGR)
        if 'DeepGlobe' in dataset:
            orig_img = cv2.resize(orig_img, (orig_img.shape[0] // 4, orig_img.shape[1] // 4))
            pred_segmask = cv2.resize(pred_segmask, (pred_segmask.shape[0] // 4, pred_segmask.shape[1] // 4),
                                      interpolation=cv2.INTER_NEAREST)
        imgio.imsave(os.path.join(out_dir, filename.replace('.jpg', '') + '.png'), pred_segmask / 256.0)
        imgio.imsave(os.path.join(out_dir, filename.replace('.jpg', '') + '_overlay.png'),
                     (1 - OVERLAY_R) * orig_img / 256.0 + OVERLAY_R * pred_segmask / 256.0)
        if is_verbose:
            print('\t\tElapsed time (s): %s' % (time.time() - start_time))
    mIoU = np.mean(intersects / (unions + 1e-7))
    df = pd.DataFrame({'Class': seg_class_names + ['Mean'], 'IoU': list(intersects / (unions + 1e-7)) + [mIoU]},
                      columns=['Class', 'IoU'])
    xlsx_path = os.path.join(eval_cues_dir, 'metrics_' + sess_id + '_' + eval_set + '.xlsx')
    df.to_excel(xlsx_path)

def eval_cues_adp(dataset, model_type, sess_id, batch_size, size, set_name, is_verbose):
    ac = ADPCues(dataset + '_' + model_type, batch_size, size, model_dir=MODEL_ROOT)
    seed_size = 41
    OVERLAY_R = 0.75

    # Load network and thresholds
    out_dirs = {}
    out_dirs['morph'] = os.path.join(OUT_ROOT, sess_id, 'morph')
    if not os.path.exists(out_dirs['morph']):
        os.makedirs(out_dirs['morph'])
    out_dirs['func'] = os.path.join(OUT_ROOT, sess_id, 'func')
    if not os.path.exists(out_dirs['func']):
        os.makedirs(out_dirs['func'])
    ac.build_model()

    # Load images
    if not os.path.exists('data'):
        os.makedirs('data')
    if is_verbose:
        print('\tGetting Grad-CAM weights for given network')
    alpha = ac.get_grad_cam_weights(np.zeros((1, size, size, 3)))

    # Read in image names
    img_names = ac.get_img_names(set_name)

    # Process images in batches
    n_batches = len(img_names) // ac.batch_size + 1
    for iter_batch in range(n_batches):
        start_time = time.time()
        if is_verbose:
            print('\tBatch #%d of %d' % (iter_batch+1, n_batches))
        start_idx = iter_batch * ac.batch_size
        end_idx = min(start_idx + ac.batch_size - 1, len(img_names) - 1)
        cur_batch_sz = end_idx - start_idx + 1
        img_batch_norm, img_batch = ac.read_batch(img_names[start_idx:end_idx + 1])
        # Determine passing classes
        predicted_scores = ac.model.predict(img_batch_norm)
        is_pass_threshold = np.greater_equal(predicted_scores, ac.thresholds)

        # Generate Grad-CAM
        H = ac.grad_cam(alpha, img_batch_norm, is_pass_threshold)
        H = np.transpose(H, (0, 3, 1, 2))
        H = resize_stack(H, (seed_size, seed_size))

        # Split Grad-CAM into {morph, func}
        H_split = {}
        H_split['morph'], H_split['func'] = ac.split_by_httclass(H)
        is_pass = {}
        is_pass['morph'], is_pass['func'] = ac.split_by_httclass(is_pass_threshold)

        # Modify Grad-CAM for each HTT type separately
        seeds = {}
        for htt_class in ['morph', 'func']:
            seeds[htt_class] = np.zeros((cur_batch_sz, len(ac.classes['valid_' + htt_class]), seed_size, seed_size))
            seeds[htt_class][:, ac.classinds[htt_class + '2valid']] = H[:, ac.classinds['all2' + htt_class]]
            class_inds = [ac.classinds_arr[htt_class + '2valid'][is_pass[htt_class][i]] for i in range(cur_batch_sz)]

            # Modify heatmaps
            if htt_class == 'morph':
                seeds[htt_class] = ac.modify_by_htt(seeds[htt_class], img_batch, ac.classes['valid_' + htt_class])
            elif htt_class == 'func':
                class_inds = [np.append(1, x) for x in class_inds]
                adipose_inds = [i for i, x in enumerate(ac.classes['morph']) if x in ['A.W', 'A.B', 'A.M']]
                gradcam_adipose = seeds['morph'][:, adipose_inds]
                seeds[htt_class] = ac.modify_by_htt(seeds[htt_class], img_batch, ac.classes['valid_' + htt_class],
                                                 gradcam_adipose=gradcam_adipose)

            # Evaluate
            ac.update_cues(seeds[htt_class], class_inds, htt_class, list(range(start_idx, end_idx + 1)))
            gt_batch = ac.read_gt_batch(htt_class, img_names[start_idx:end_idx + 1])

            for j in range(cur_batch_sz):
                pred_segmask = np.zeros((size, size, 3))
                gt_r = gt_batch[j, :, :, 0]
                gt_g = gt_batch[j, :, :, 1]
                gt_b = gt_batch[j, :, :, 2]
                cues_i = ac.cues[htt_class]['%s_cues' % (start_idx + j)]
                cues = np.zeros((seed_size, seed_size, len(ac.colours[htt_class])))
                cues[cues_i[1], cues_i[2], cues_i[0]] = 1.0
                for k, gt_colour in enumerate(ac.colours[htt_class]):
                    gt_mask = (gt_r == gt_colour[0]) & (gt_g == gt_colour[1]) & (gt_b == gt_colour[2])
                    pred_mask = cv2.resize(cues[:, :, k], (size, size), interpolation=cv2.INTER_NEAREST) == 1.0
                    pred_segmask += np.expand_dims(pred_mask, axis=2) * \
                                    np.expand_dims(np.expand_dims(gt_colour, axis=0), axis=0)
                    ac.intersects[htt_class][k] += np.sum(gt_mask & pred_mask)
                    ac.unions[htt_class][k] += np.sum(gt_mask | pred_mask)
                imgio.imsave(os.path.join(out_dirs[htt_class], os.path.splitext(img_names[start_idx+j])[0] + '.png'), pred_segmask / 256.0)
                imgio.imsave(os.path.join(out_dirs[htt_class], os.path.splitext(img_names[start_idx+j])[0] + '_overlay.png'),
                             (1-OVERLAY_R) * img_batch[j] / 256.0 + OVERLAY_R * pred_segmask / 256.0)

        elapsed_time = time.time() - start_time
        if is_verbose:
            print('\t\tElapsed time: %s seconds (%s seconds/image)' % (elapsed_time, elapsed_time / cur_batch_sz))

    iou = {}
    miou = {}
    for htt_class in ['morph', 'func']:
        iou[htt_class] = ac.intersects[htt_class] / ac.unions[htt_class]
        miou[htt_class] = np.mean(iou[htt_class])
        if is_verbose:
            print('\tmIoU (%s): %s' % (htt_class, miou[htt_class]))

        eval_dir = os.path.join(EVAL_CUES_ROOT, sess_id)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        df = pd.DataFrame({'Class': ac.classes['valid_' + htt_class] + ['Mean'], 'IoU': list(iou[htt_class]) + [miou[htt_class]]},
                          columns=['Class', 'IoU'])
        xlsx_path = os.path.join(eval_dir, 'metrics_ADP-' + htt_class + '_' + set_name + '_' + model_type + '.xlsx')
        df.to_excel(xlsx_path)

if __name__ == "__main__":
    # ADP
    gen_cues(dataset='ADP', model_type='VGG16', batch_size=16, is_verbose=True)
    eval_cues(dataset='ADP', model_type='VGG16', batch_size=16, set_name='tuning', is_verbose=True)
    eval_cues(dataset='ADP', model_type='VGG16', batch_size=16, set_name='segtest', is_verbose=True)
    gen_cues(dataset='ADP', model_type='X1.7', batch_size=16, is_verbose=True)
    eval_cues(dataset='ADP', model_type='X1.7', batch_size=16, set_name='tuning', is_verbose=True)
    eval_cues(dataset='ADP', model_type='X1.7', batch_size=16, set_name='segtest', is_verbose=True)

    # PASCAL VOC 2012
    gen_cues(dataset='VOC2012', model_type='VGG16', batch_size=8, is_verbose=True) # validated
    eval_cues(dataset='VOC2012', model_type='VGG16', batch_size=8, is_verbose=True) # validated
    gen_cues(dataset='VOC2012', model_type='M7', batch_size=8, is_verbose=True)
    eval_cues(dataset='VOC2012', model_type='M7', batch_size=8, is_verbose=True)

    # DeepGlobe
    gen_cues(dataset='DeepGlobe_train75', model_type='VGG16', batch_size=8, is_verbose=True)
    eval_cues(dataset='DeepGlobe_train75', model_type='VGG16', batch_size=8, is_verbose=True)
    gen_cues(dataset='DeepGlobe_train75', model_type='M7', batch_size=8, is_verbose=True)
    eval_cues(dataset='DeepGlobe_train75', model_type='M7', batch_size=8, is_verbose=True)
    gen_cues(dataset='DeepGlobe_train37.5', model_type='VGG16', batch_size=8, is_verbose=True)
    eval_cues(dataset='DeepGlobe_train37.5', model_type='VGG16', batch_size=8, is_verbose=True)
    gen_cues(dataset='DeepGlobe_train37.5', model_type='M7', batch_size=8, is_verbose=True)
    eval_cues(dataset='DeepGlobe_train37.5', model_type='M7', batch_size=8, is_verbose=True)