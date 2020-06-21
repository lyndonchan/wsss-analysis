import os
import configparser
import math
import keras
from keras import optimizers
from keras.models import model_from_json
import scipy.io as sio

from clr_callback import *
from epoch_callback import *
from utilities import *

config = configparser.ConfigParser()
config.read('../settings.ini')
DATA_ROOT = os.path.join(config['Download Directory']['data_dir'])
MODEL_ROOT = os.path.join(config['Download Directory']['data_dir'], config['Data Folders']['model_cnn_dir'])
LOG_ROOT = './log'
CKPT_ROOT = './ckpt'
EVAL_ROOT = './eval'

def train(dataset, model_type, epochs, batch_size, should_clr=True, should_pretrained=True, should_wgt=True,
          should_backup=False, should_reset=False):
    assert dataset in ['ADP', 'VOC2012', 'DeepGlobe', 'DeepGlobe_balanced']
    assert model_type in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'X1.7', 'VGG16fg', 'VGG16fg_bn']
    assert epochs > 0
    assert batch_size > 0

    if model_type in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'X1.7']:
        img_size = 224
    elif model_type in ['VGG16fg', 'VGG16fg_bn']:
        img_size = 321
    if should_clr:
        clr_base_lr = 1e-3
        clr_max_lr = 2e-2
        clr_spc = 4
    base_lr = 1e-3
    lr_dropstep = 20
    lr_droprate = 0.5
    momentum = 0.9

    sess_id = dataset + '_' + model_type.replace('fg', '').replace('_bn', '')
    if not should_pretrained:
        sess_id += '_npt'
    log_dir = os.path.join(LOG_ROOT, sess_id)
    model_dir = os.path.join(MODEL_ROOT, sess_id)
    ckpt_dir = os.path.join(CKPT_ROOT, sess_id)
    # Overwrite old files
    if should_backup:
        backup([log_dir, model_dir, ckpt_dir])
    if should_reset:
        reset([log_dir, model_dir, ckpt_dir])
    makedir_if_nexist([log_dir, model_dir, ckpt_dir])
    print('Sess: ' + sess_id)

    # Load data and classes
    gen_train, gen_val, gen_test, classes = load_data(DATA_ROOT, dataset, model_type, [img_size, img_size, 3], batch_size)
    # Load model
    model = load_model(model_type, [img_size, img_size, 3], len(classes), use_pretrained=should_pretrained)
    # Compiling model
    opt = optimizers.SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy', f1])

    # Save model architecture
    arch_path = os.path.join(model_dir, sess_id + '.json')
    with open(arch_path, 'w') as f:
        f.write(model.to_json())
    # Callbacks
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    filepath = os.path.join(ckpt_dir, sess_id + '-{epoch_num:02d}.hdf5')
    ckpt_cb = keras.callbacks.ModelCheckpoint(filepath, monitor='epoch_num', verbose=0,
                                              save_best_only=True, save_weights_only=False, mode='max',
                                              period=lr_dropstep)

    # Resume training, if previous checkpoint exists
    latest_checkpoint, latest_epoch = find_latest_checkpoint(ckpt_dir, sess_id)
    if latest_checkpoint is not None:
        print('Loading model from ' + latest_checkpoint)
        model = keras.models.load_model(latest_checkpoint, custom_objects={'f1': f1})
    if should_wgt:
        class_weights = list(gen_train.n / (np.sum(gen_train.data, axis=0) + 1e-7))
    else:
        class_weights = list(np.ones_like(np.sum(gen_train.data, axis=0)))

    # Train
    if should_clr:
        step_sz = lr_dropstep / clr_spc * gen_train.n // batch_size
        if latest_epoch is not None:
            base_lr = clr_base_lr * lr_droprate ** math.ceil(latest_epoch / lr_dropstep)
            max_lr = clr_max_lr * lr_droprate ** math.ceil(latest_epoch / lr_dropstep)
            clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_sz, completed_epochs=latest_epoch)
            clr_sess_rng = range(math.ceil(latest_epoch / lr_dropstep), math.ceil(epochs / lr_dropstep))
        else:
            clr = CyclicLR(base_lr=clr_base_lr, max_lr=clr_max_lr, step_size=step_sz, completed_epochs=latest_epoch)
            clr_sess_rng = range(math.ceil(epochs / lr_dropstep))
        for iter_clr_sess in clr_sess_rng:
            if (iter_clr_sess + 1) * lr_dropstep <= epochs:
                curr_epochs = lr_dropstep
            else:
                curr_epochs = epochs - (iter_clr_sess - 1) * lr_dropstep
            model.fit_generator(generator=gen_train,
                                steps_per_epoch=gen_train.n // batch_size,
                                validation_data=gen_val,
                                validation_steps=gen_val.n // batch_size,
                                epochs=curr_epochs,
                                callbacks=[clr, tensorboard_cb, ckpt_cb],  # , epoch_cb],
                                verbose=2,
                                class_weight=class_weights)
            curr_base_lr = clr_base_lr * lr_droprate ** (iter_clr_sess + 1)
            curr_max_lr = clr_max_lr * lr_droprate ** (iter_clr_sess + 1)
            clr._reset(new_base_lr=curr_base_lr, new_max_lr=curr_max_lr, new_step_size=step_sz)
    else:
        def lr_scheduler(epoch):
            return base_lr * (lr_droprate ** ((epoch + latest_epoch) // lr_dropstep))

        lr_reduce_cb = keras.callbacks.LearningRateScheduler(lr_scheduler)
        epoch_cb = EpochCallback(completed_epochs=latest_epoch)
        model.fit_generator(generator=gen_train,
                            steps_per_epoch=gen_train.n // batch_size,
                            validation_data=gen_val,
                            validation_steps=gen_val.n // batch_size,
                            epochs=epochs - latest_epoch,
                            callbacks=[lr_reduce_cb, tensorboard_cb, epoch_cb, ckpt_cb],  # put epoch_cb before ckpt_cb!
                            verbose=2,
                            class_weight=class_weights)
    # Save trained weights
    weights_path = os.path.join(model_dir, sess_id + '.h5')
    model.save_weights(weights_path)
    
def predict(dataset, model_type, batch_size, should_pretrained=True, should_backup=False, should_reset=False):
    assert dataset in ['ADP', 'VOC2012', 'DeepGlobe', 'DeepGlobe_balanced']
    assert model_type in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'X1.7', 'VGG16fg', 'VGG16fg_bn']
    assert batch_size > 0

    if model_type in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'X1.7']:
        img_size = 224
    elif model_type in ['VGG16fg', 'VGG16fg_bn']:
        img_size = 321

    sess_id = dataset + '_' + model_type.replace('fg', '').replace('_bn', '')
    if not should_pretrained:
        sess_id += '_npt'
    model_dir = os.path.join(MODEL_ROOT, sess_id)
    eval_dir = os.path.join(EVAL_ROOT, sess_id)
    # Overwrite old files
    if should_backup:
        backup([eval_dir])
    if should_reset:
        reset([eval_dir])
    makedir_if_nexist([eval_dir])
    print('Sess: ' + sess_id)

    # Load model
    arch_path = os.path.join(model_dir, sess_id + '.json')
    with open(arch_path, 'r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    weights_path = os.path.join(model_dir, sess_id + '.h5')
    model.load_weights(weights_path)

    # Load data and classes
    print('Loading data')
    gen_train, gen_val, gen_test, classes = load_data(DATA_ROOT, dataset, model_type, [img_size, img_size, 3], batch_size)
    # Compiling model
    opt = optimizers.SGD(lr=0.0, momentum=0.0, decay=0.0, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

    # Predict
    gen_train.shuffle = False
    print('Predicting on validation set')
    pred_valid = model.predict_generator(gen_val, steps=len(gen_val))
    print('Predicting on test set')
    pred_test = model.predict_generator(gen_test, steps=len(gen_test))

    if model_type == 'X1.7' and dataset == 'ADP':
        CLASS_NAMES_AUG = ['E', 'E.M', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C',
                           'C.D', 'C.D.I', 'C.D.R', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.M.C', 'S.M.S',
                           'S.E', 'S.C', 'S.C.H', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K', 'N', 'N.P',
                           'N.R', 'N.R.B', 'N.R.A', 'N.G', 'N.G.M', 'N.G.A', 'N.G.O', 'N.G.E', 'N.G.R', 'N.G.W',
                           'N.G.T', 'G', 'G.O', 'G.N', 'T']
        m_idx = [i for i, x in enumerate(CLASS_NAMES_AUG) if x in classes]
        pred_valid = pred_valid[:, m_idx]
        pred_test = pred_test[:, m_idx]

    # Get ROC analysis
    # - Get optimal class thresholds
    print('Getting optimal thresholds')
    class_thresholds, _, _ = get_optimal_thresholds(gen_val.data, pred_valid)
    # class_thresholds = [0.5] * len(classes)
    adict = {}
    adict['optimalScoreThresh'] = class_thresholds
    thresh_path = os.path.join(model_dir, sess_id + '.mat')
    sio.savemat(thresh_path, adict)

    # - Get thresholded class accuracies
    print('Evaluating TPR, FPR, TNR, FNR, ACC, F1')
    if dataset in ['ADP'] or 'DeepGlobe' in dataset:
        eval_generator = gen_test
        eval_pred = pred_test
    elif dataset in ['VOC2012']:
        eval_generator = gen_val
        eval_pred = pred_valid
    metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s = \
        get_thresholded_metrics_class(eval_generator.data, eval_pred, class_thresholds)
    mean_tpr, mean_fpr, mean_tnr, mean_fnr, mean_acc, mean_f1 = \
        get_thresholded_metrics_overall(eval_generator.data, eval_pred, class_thresholds)
    # - Plot ROC curves
    print('Plotting ROC')
    _, class_fprs, class_tprs = get_optimal_thresholds(eval_generator.data, eval_pred)
    plot_rocs(class_fprs, class_tprs, classes, eval_dir, sess_id)
    # - Write metrics to Excel
    print('Writing to Excel')
    write_to_excel(metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s,
                   mean_tpr, mean_fpr, mean_tnr, mean_fnr, mean_acc, mean_f1, classes, eval_dir, sess_id)

if __name__ == "__main__":
    # ADP
    train(dataset='ADP', model_type='VGG16fg', epochs=80, batch_size=16, should_clr=False)
    predict(dataset='ADP', model_type='VGG16fg', batch_size=16)
    train(dataset='ADP', model_type='X1.7', epochs=80, batch_size=16, should_clr=False)
    predict(dataset='ADP', model_type='X1.7', batch_size=16)
    # VOC2012
    train(dataset='VOC2012', model_type='VGG16fg_bn', epochs=80, batch_size=8, should_clr=False)
    predict(dataset='VOC2012', model_type='VGG16fg_bn', batch_size=8)
    train(dataset='VOC2012', model_type='M7', epochs=80, batch_size=8, should_clr=False)
    predict(dataset='VOC2012', model_type='M7', batch_size=8)
    # DeepGlobe
    train(dataset='DeepGlobe', model_type='VGG16fg_bn', epochs=80, batch_size=8, should_clr=False)
    predict(dataset='DeepGlobe', model_type='VGG16fg_bn', batch_size=8)
    train(dataset='DeepGlobe', model_type='M7', epochs=80, batch_size=8, should_clr=False)
    predict(dataset='DeepGlobe', model_type='M7', batch_size=8)
    # DeepGlobe_balanced
    train(dataset='DeepGlobe_balanced', model_type='VGG16fg_bn', epochs=80, batch_size=8, should_clr=False)
    predict(dataset='DeepGlobe_balanced', model_type='VGG16fg_bn', batch_size=8)
    train(dataset='DeepGlobe_balanced', model_type='M7', epochs=80, batch_size=8, should_clr=False)
    predict(dataset='DeepGlobe_balanced', model_type='M7', batch_size=8)