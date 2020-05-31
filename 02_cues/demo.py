import pickle
import scipy.io as sio
import skimage.io as imgio
from keras import optimizers
from keras.models import model_from_json
import pandas as pd
from tqdm import tqdm
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

def gen_cues(dataset, model_type, batch_size, is_wpt=True, set_name=None, run_train=True, is_verbose=True, first_inds=None):
    """Generate weak segmentation cues for VOC2012 and DeepGlobe datasets, with redirect for ADP

    Parameters
    ----------
    dataset : str
        The name of the dataset (i.e. 'ADP', 'VOC2012', 'DeepGlobe_train75', or 'DeepGlobe_balanced')
    model_type : str
    The name of the model to use for generating cues (i.e. 'X1.7', 'M7', 'M7bg', 'VGG16', 'VGG16bg')
    batch_size : int
        The batch size (>0)
    set_name : str, optional
        The name of the name of the evaluation set, if ADP (i.e. 'tuning' or 'segtest')
    run_train : bool, optional
        Whether to run on the training set
    is_verbose : bool, optional
        Whether to activate message verbosity
    """
    assert(dataset in ['ADP', 'VOC2012', 'DeepGlobe', 'DeepGlobe_balanced'])
    assert(model_type in ['X1.7', 'M7', 'VGG16'])
    assert(is_wpt in [True, False])
    if is_wpt:
        pt_str = '_wpt'
    else:
        pt_str = '_npt'
    assert(os.path.exists(os.path.join(MODEL_ROOT, dataset + '_' + model_type + pt_str)))
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

    model_dir = os.path.join(MODEL_ROOT, dataset + '_' + model_type + pt_str)
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
        # Redirect to helper function if ADP
        if run_train:
            gen_cues_adp(model_type, is_wpt, batch_size, img_size, train_cues_dir, set_name, is_verbose, first_inds)
        else:
            gen_cues_adp(model_type, is_wpt, batch_size, img_size, eval_cues_dir, set_name, is_verbose, first_inds)
        return
    ds = Dataset(data_type=dataset, size=img_size, batch_size=batch_size, first_inds=first_inds)
    if run_train:
        gen_curr = ds.set_gens[ds.sets[ds.is_evals.index(False)]]
    else:
        gen_curr = ds.set_gens[ds.sets[ds.is_evals.index(True)]]
    img_names = gen_curr.filenames

    # Load and compile models
    def load_model(model_dir, model_name):
        # Load model
        arch_path = os.path.join(model_dir, model_name + '.json')
        with open(arch_path, 'r') as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        weights_path = os.path.join(model_dir, model_name + '.h5')
        model.load_weights(weights_path)
        # Compile model
        opt = optimizers.SGD(lr=0.0, momentum=0.0, decay=0.0, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
        # Get alpha
        final_layer = find_final_layer(model)
        alpha = get_grad_cam_weights(model, final_layer, np.zeros((1, img_size, img_size, 3)))
        # Get thresholds
        thresh_path = os.path.join(model_dir, model_name + '.mat')
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
            fgbg_dir[fgbg_mode] = os.path.join(MODEL_ROOT, dataset + '_' + model_type + pt_str)
            fgbg_sess[fgbg_mode] = sess_id + pt_str
        elif fgbg_mode == 'bg':
            fgbg_dir[fgbg_mode] = os.path.join(MODEL_ROOT, dataset + '_' + model_type + 'bg' + pt_str)
            fgbg_sess[fgbg_mode] = sess_id + 'bg' + pt_str
        model[fgbg_mode], alpha[fgbg_mode], final_layer[fgbg_mode], thresholds[fgbg_mode] = \
            load_model(fgbg_dir[fgbg_mode], fgbg_sess[fgbg_mode])

    # Process by batch
    n_batches = math.ceil(len(img_names) / batch_size)
    for iter_batch in tqdm(range(n_batches)):
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
        # Read batches of normalized/unnormalized images
        img_batch_norm, img_batch = read_batch(gen_curr.directory, img_names[start_idx:end_idx + 1], cur_batch_sz,
                                               (img_size, img_size), img_mean=dataset_mean, img_std=dataset_std)
        for fgbg_mode in fgbg_modes:
            # Determine passing classes
            pred_scores = model[fgbg_mode].predict(img_batch_norm)
            keep_inds = np.arange(len(ds.class_names))
            if ignore_ind is not None:
                keep_inds = np.delete(keep_inds, ignore_ind)
                pred_scores = pred_scores[:, keep_inds]
                thresholds[fgbg_mode] = thresholds[fgbg_mode][:, keep_inds]
            is_pass_threshold[fgbg_mode] = np.greater_equal(pred_scores, thresholds[fgbg_mode]) * \
                                           gen_curr.data[start_idx:end_idx+1, keep_inds]

            # Generate Grad-CAM
            if fgbg_mode == 'fg':
                H[fgbg_mode] = grad_cam(model[fgbg_mode], alpha[fgbg_mode], img_batch_norm, is_pass_threshold[fgbg_mode],
                                        final_layer[fgbg_mode], keep_inds, [img_size, img_size])
            elif fgbg_mode == 'bg':
                H[fgbg_mode] = grad_cam(model[fgbg_mode], alpha[fgbg_mode], img_batch_norm, is_pass_threshold[fgbg_mode],
                                        final_layer[fgbg_mode], keep_inds, [img_size, img_size])
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
    if is_verbose:
        print('Saving localization cues')
    if run_train:
        # Training set localization cues (for further processing/training in 02, 03)
        pickle.dump(localization_cues, open(os.path.join(train_cues_dir, 'localization_cues.pickle'), 'wb'))
    else:
        # Validation set localization cues (for evaluation only)
        pickle.dump(localization_cues, open(os.path.join(eval_cues_dir, 'localization_cues_val.pickle'), 'wb'))

def gen_cues_adp(model_type, is_wpt, batch_size, size, cues_dir, set_name, is_verbose, first_inds):
    """Generate weak segmentation cues for ADP (helper function)

    Parameters
    ----------
    model_type : str
        The name of the model to use for generating cues (i.e. 'X1.7', 'M7', 'M7bg', 'VGG16', 'VGG16bg')
    batch_size : int
        The batch size (>0)
    size : int
        The length of the resized input image
    cues_dir : str
        The directory to save the cues to
    set_name : str
        The name of the name of the evaluation set (i.e. 'tuning' or 'segtest')
    is_verbose : bool, optional
        Whether to activate message verbosity
    """
    if is_wpt:
        pt_str = '_wpt'
    else:
        pt_str = '_npt'
    ac = ADPCues('ADP_' + model_type + pt_str, batch_size, size, model_dir=MODEL_ROOT, first_inds=first_inds)
    seed_size = 41

    # Load network and thresholds
    cues_dirs = {}
    for htt_class in ['morph', 'func']:
        cues_dirs[htt_class] = os.path.join(cues_dir, htt_class)
        makedir_if_nexist([cues_dirs[htt_class]])
    ac.build_model()

    # Load Grad-CAM weights
    if not os.path.exists('data'):
        os.makedirs('data')
    if is_verbose:
        print('\tGetting Grad-CAM weights for given network')
    alpha = ac.get_grad_cam_weights(np.zeros((1, size, size, 3)))

    # Read in image names
    img_names = ac.get_img_names(set_name)

    # Process images in batches
    n_batches = len(img_names) // batch_size + 1
    for iter_batch in tqdm(range(n_batches)):
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

            # Update localization cues
            ac.update_cues(seeds[htt_class], class_inds, htt_class, list(range(start_idx, end_idx + 1)))
    # Save localization cues
    if is_verbose:
        print('\tSaving localization cues')
    pickle.dump(ac.cues['morph'], open(os.path.join(cues_dirs['morph'], 'localization_cues.pickle'), 'wb'))
    pickle.dump(ac.cues['func'], open(os.path.join(cues_dirs['func'], 'localization_cues.pickle'), 'wb'))

def eval_cues(dataset, model_type, batch_size, is_wpt=True, set_name=None, run_train=False, should_saveimg=True,
              is_verbose=True, first_inds=None):
    """Evaluate weak segmentation cues for VOC2012 and DeepGlobe datasets, with redirect for ADP

    Parameters
    ----------
    dataset : str
        The name of the dataset (i.e. 'ADP', 'VOC2012', 'DeepGlobe', or 'DeepGlobe_balanced')
    model_type : str
        The name of the model to use for generating cues (i.e. 'X1.7', 'M7', 'M7bg', 'VGG16', 'VGG16bg')
    batch_size : int
        The batch size (>0)
    set_name : str, optional
        The name of the name of the evaluation set, if ADP (i.e. 'tuning' or 'segtest')
    run_train : bool, optional
        Whether to run on the training set
    should_saveimg : bool, optional
        Whether to save debug images
    is_verbose : bool, optional
        Whether to activate message verbosity
    """
    assert(dataset in ['ADP', 'VOC2012', 'DeepGlobe', 'DeepGlobe_balanced'])
    assert(model_type in ['X1.7', 'M7', 'M7bg', 'VGG16', 'VGG16bg'])
    assert(is_wpt in [True, False])
    if is_wpt:
        pt_str = '_wpt'
    else:
        pt_str = '_npt'
    assert(os.path.exists(os.path.join(MODEL_ROOT, dataset + '_' + model_type + pt_str)))
    assert(batch_size > 0)
    assert(set_name in [None, 'tuning', 'segtest'])
    assert(type(run_train) is bool)
    assert(type(should_saveimg) is bool)
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
    if should_saveimg:
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
        # Redirect to helper function if ADP
        eval_cues_adp(model_type, sess_id, is_wpt, batch_size, img_size, set_name, should_saveimg, is_verbose, first_inds)
        return
    makedir_if_nexist([eval_cues_dir])
    ds = Dataset(data_type=dataset, size=img_size, batch_size=batch_size, first_inds=first_inds)
    eval_set = ds.sets[ds.is_evals.index(True)]
    gen_eval = ds.set_gens[eval_set]
    colours = get_colours(dataset)

    # Load localization cues from validation set for evaluation purposes
    cues_path = os.path.join(eval_cues_dir, 'localization_cues_val.pickle')
    if not os.path.exists(cues_path):
        # Generate first if not already existing
        gen_cues(dataset, model_type, batch_size, run_train=False, is_verbose=is_verbose, first_inds=first_inds)
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

    # Evaluate one image at a time
    for iter_file, filename in enumerate(tqdm(gen_eval.filenames)):
        if dataset == 'VOC2012':
            # Load GT segmentation
            gt_filepath = os.path.join(gt_dir, filename.replace('.jpg', '.png'))
            gt_idx = cv2.cvtColor(cv2.imread(gt_filepath), cv2.COLOR_RGB2BGR)[:, :, 0]
            # Load predicted segmentation
            cues_i = localization_cues['%s_cues' % iter_file]
            cues_pred = np.zeros([seed_size, seed_size, len(colours)])
            cues_pred[cues_i[1], cues_i[2], cues_i[0]] = 1.0
            cues_pred_max = np.argmax(cues_pred, axis=-1)
            pred_idx = cv2.resize(np.uint8(cues_pred_max), (gt_idx.shape[1], gt_idx.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            pred_segmask = np.zeros((gt_idx.shape[0], gt_idx.shape[1], 3))
            # Evaluate predicted segmentation against GT
            for k in range(len(colours)):
                intersects[k] += np.sum((gt_idx == k) & (pred_idx == k))
                unions[k] += np.sum((gt_idx == k) | (pred_idx == k))
                pred_segmask += np.expand_dims(pred_idx == k, axis=2) * \
                                np.expand_dims(np.expand_dims(colours[k], axis=0), axis=0)
        elif 'DeepGlobe' in dataset:
            # Load GT segmentation
            gt_filepath = os.path.join(gt_dir, filename.replace('.jpg', '.png'))
            gt_curr = cv2.cvtColor(cv2.imread(gt_filepath), cv2.COLOR_RGB2BGR)
            # Load predicted segmentation
            cues_i = localization_cues['%s_cues' % iter_file]
            cues_pred = np.zeros([seed_size, seed_size, len(colours)])
            cues_pred[cues_i[1], cues_i[2], cues_i[0]] = 1.0
            cues_pred_max = np.argmax(cues_pred, axis=-1)
            cues_pred_max[np.sum(cues_pred, axis=-1) == 0] = ignore_ind
            pred_idx = cv2.resize(np.uint8(cues_pred_max), (gt_curr.shape[1], gt_curr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            pred_segmask = np.zeros((gt_curr.shape[0], gt_curr.shape[1], 3))
            # Evaluate predicted segmentation against GT
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
        # Save debugging images to file
        if should_saveimg:
            orig_filepath = os.path.join(gen_eval.directory, filename)
            orig_img = cv2.cvtColor(cv2.imread(orig_filepath), cv2.COLOR_RGB2BGR)
            # Downsample to save space if DeepGlobe
            if 'DeepGlobe' in dataset:
                orig_img = cv2.resize(orig_img, (orig_img.shape[0] // 4, orig_img.shape[1] // 4))
                pred_segmask = cv2.resize(pred_segmask, (pred_segmask.shape[0] // 4, pred_segmask.shape[1] // 4),
                                          interpolation=cv2.INTER_NEAREST)
            imgio.imsave(os.path.join(out_dir, filename.replace('.jpg', '') + '.png'), np.uint8(pred_segmask),
                         check_contrast=False)
            imgio.imsave(os.path.join(out_dir, filename.replace('.jpg', '') + '_overlay.png'),
                         np.uint8((1 - OVERLAY_R) * orig_img + OVERLAY_R * pred_segmask), check_contrast=False)
    # Calculate mIoU and save to .xlsx metrics file
    mIoU = np.mean(intersects / (unions + 1e-7))
    df = pd.DataFrame({'Class': seg_class_names + ['Mean'], 'IoU': list(intersects / (unions + 1e-7)) + [mIoU]},
                      columns=['Class', 'IoU'])
    xlsx_path = os.path.join(eval_cues_dir, 'metrics_' + sess_id + '_' + eval_set + '.xlsx')
    df.to_excel(xlsx_path)

def eval_cues_adp(model_type, sess_id, is_wpt, batch_size, size, set_name, should_saveimg, is_verbose, first_inds):
    """Evaluate weak segmentation cues for ADP (helper function)

    Parameters
    ----------
    model_type : str
        The name of the model to use for generating cues (i.e. 'X1.7', 'M7', 'M7bg', 'VGG16', 'VGG16bg')
    sess_id : str
        The identifying string for the current session
    batch_size : int
        The batch size (>0)
    size : int
        The length of the resized input image
    set_name : str, optional
        The name of the name of the evaluation set, if ADP (i.e. 'tuning' or 'segtest')
    should_saveimg : bool, optional
        Whether to save debug images
    is_verbose : bool, optional
        Whether to activate message verbosity
    """
    if is_wpt:
        pt_str = '_wpt'
    else:
        pt_str = '_npt'
    ac = ADPCues('ADP_' + model_type + pt_str, batch_size, size, model_dir=MODEL_ROOT, first_inds=first_inds)
    seed_size = 41
    OVERLAY_R = 0.75

    # Load network and thresholds
    if should_saveimg:
        out_dirs = {}
        for htt_class in ['morph', 'func']:
            out_dirs[htt_class] = os.path.join(OUT_ROOT, sess_id, htt_class)
            makedir_if_nexist([out_dirs[htt_class]])
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
    for iter_batch in tqdm(range(n_batches)):
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
            # Update cues
            ac.update_cues(seeds[htt_class], class_inds, htt_class, list(range(start_idx, end_idx + 1)))

            # Load GT segmentation images
            gt_batch = ac.read_gt_batch(htt_class, img_names[start_idx:end_idx + 1])
            # Process images one at a time
            for j in range(cur_batch_sz):
                # Separate GT segmentation images into R, G, B channels
                gt_r = gt_batch[j, :, :, 0]
                gt_g = gt_batch[j, :, :, 1]
                gt_b = gt_batch[j, :, :, 2]
                # Load predicted segmentations
                cues_i = ac.cues[htt_class]['%s_cues' % (start_idx + j)]
                cues = np.zeros((seed_size, seed_size, len(ac.colours[htt_class])))
                cues[cues_i[1], cues_i[2], cues_i[0]] = 1.0
                pred_segmask = np.zeros((size, size, 3))
                # Evaluate predicted segmentations
                for k, gt_colour in enumerate(ac.colours[htt_class]):
                    gt_mask = (gt_r == gt_colour[0]) & (gt_g == gt_colour[1]) & (gt_b == gt_colour[2])
                    pred_mask = cv2.resize(cues[:, :, k], (size, size), interpolation=cv2.INTER_NEAREST) == 1.0
                    pred_segmask += np.expand_dims(pred_mask, axis=2) * \
                                    np.expand_dims(np.expand_dims(gt_colour, axis=0), axis=0)
                    ac.intersects[htt_class][k] += np.sum(gt_mask & pred_mask)
                    ac.unions[htt_class][k] += np.sum(gt_mask | pred_mask)
                # Save debugging images to file
                if should_saveimg:
                    imgio.imsave(os.path.join(out_dirs[htt_class], os.path.splitext(img_names[start_idx+j])[0] + '.png'),
                                 np.uint8(pred_segmask), check_contrast=False)
                    imgio.imsave(os.path.join(out_dirs[htt_class], os.path.splitext(img_names[start_idx+j])[0] + '_overlay.png'),
                                 np.uint8((1-OVERLAY_R) * img_batch[j] + OVERLAY_R * pred_segmask), check_contrast=False)
    # Calculate IoU, mIoU metrics
    iou = {}
    miou = {}
    for htt_class in ['morph', 'func']:
        iou[htt_class] = ac.intersects[htt_class] / ac.unions[htt_class]
        miou[htt_class] = np.mean(iou[htt_class])
        if is_verbose:
            print('\tmIoU (%s): %s' % (htt_class, miou[htt_class]))

        eval_dir = os.path.join(EVAL_CUES_ROOT, sess_id)
        makedir_if_nexist([eval_dir])
        # Save to .xlsx metrics file
        df = pd.DataFrame({'Class': ac.classes['valid_' + htt_class] + ['Mean'], 'IoU': list(iou[htt_class]) + [miou[htt_class]]},
                          columns=['Class', 'IoU'])
        xlsx_path = os.path.join(eval_dir, 'metrics_ADP-' + htt_class + '_' + set_name + '_' + model_type + '.xlsx')
        df.to_excel(xlsx_path)

if __name__ == "__main__":
    # ADP
    # gen_cues(dataset='ADP', model_type='VGG16', batch_size=16, is_verbose=True, first_inds=100)
    # eval_cues(dataset='ADP', model_type='VGG16', batch_size=16, set_name='tuning', should_saveimg=True, is_verbose=True, first_inds=100)
    # eval_cues(dataset='ADP', model_type='VGG16', batch_size=16, set_name='segtest', should_saveimg=True, is_verbose=True, first_inds=100)
    # gen_cues(dataset='ADP', model_type='X1.7', batch_size=16, is_verbose=True, first_inds=100)
    # eval_cues(dataset='ADP', model_type='X1.7', batch_size=16, set_name='tuning', should_saveimg=True, is_verbose=True, first_inds=100)
    # eval_cues(dataset='ADP', model_type='X1.7', batch_size=16, set_name='segtest', should_saveimg=True, is_verbose=True, first_inds=100)

    # PASCAL VOC 2012
    gen_cues(dataset='VOC2012', model_type='VGG16', batch_size=8, is_verbose=True, first_inds=100)
    eval_cues(dataset='VOC2012', model_type='VGG16', batch_size=8, should_saveimg=True, is_verbose=True, first_inds=100)
    gen_cues(dataset='VOC2012', model_type='M7', batch_size=8, is_verbose=True, first_inds=100)
    eval_cues(dataset='VOC2012', model_type='M7', batch_size=8, should_saveimg=True, is_verbose=True, first_inds=100)

    # DeepGlobe
    gen_cues(dataset='DeepGlobe', model_type='VGG16', batch_size=8, is_verbose=True, first_inds=100)
    eval_cues(dataset='DeepGlobe', model_type='VGG16', batch_size=8, should_saveimg=True,  is_verbose=True, first_inds=100)
    gen_cues(dataset='DeepGlobe', model_type='M7', batch_size=8, is_verbose=True, first_inds=100)
    eval_cues(dataset='DeepGlobe', model_type='M7', batch_size=8, should_saveimg=True, is_verbose=True, first_inds=100)
    gen_cues(dataset='DeepGlobe_balanced', model_type='VGG16', batch_size=8, is_verbose=True, first_inds=100)
    eval_cues(dataset='DeepGlobe_balanced', model_type='VGG16', batch_size=8, should_saveimg=True, is_verbose=True, first_inds=100)
    gen_cues(dataset='DeepGlobe_balanced', model_type='M7', batch_size=8, is_verbose=True, first_inds=100)
    eval_cues(dataset='DeepGlobe_balanced', model_type='M7', batch_size=8, should_saveimg=True, is_verbose=True, first_inds=100)