import keras.backend as K
import os
import pandas as pd
from sklearn.metrics import roc_curve
import shutil
import numpy as np
import matplotlib.pyplot as plt
import model_loader
import data_loader

def load_model(model_type, size, num_classes, use_pretrained=''):
    if model_type == 'VGG16fg':
        model = model_loader.build_fg(size, num_classes)
    elif model_type == 'VGG16fg_bn':
        model = model_loader.build_VGG16fg_bn(size, num_classes)
    elif model_type == 'X1.7':
        model = model_loader.build_X1p7()
    else:
        model = model_loader.build_vgg16_experimental(size, num_classes, var_code=model_type)
    if use_pretrained:
        pretrained_weights_and_biases = model_loader.get_weights_and_bias(size)
        model = model_loader.set_weights_and_bias(model, pretrained_weights_and_biases)
    return model

def load_data(db_dir, dataset, model_type, size, batch_size):
    if dataset == 'VOC2012':
        gen_train, gen_val, gen_test, classes, train_class_counts = data_loader.load_data_VOC2012(db_dir=db_dir,
                                                                                                  size=size,
                                                                                                  batch_size=batch_size)
    elif dataset == 'ADP':
        gen_train, gen_val, gen_test, classes, train_class_counts = data_loader.load_data_ADP(db_dir=db_dir, size=size,
                                                                                              batch_size=batch_size,
                                                                                              is_aug=model_type=='X1.7')
    elif dataset in ['DeepGlobe', 'DeepGlobe_balanced']:
        gen_train, gen_val, gen_test, classes, train_class_counts = data_loader.load_data_DeepGlobe(db_dir=db_dir,
                                                                        size=size, batch_size=batch_size,
                                                                        is_balanced=dataset=='DeepGlobe_balanced')
    return gen_train, gen_val, gen_test, classes

def backup(dir_list):
    backup_dir_list = dir_list[:]
    while any([os.path.exists(x) for x in backup_dir_list]):
        backup_dir_list = [x + '-backup' for x in backup_dir_list]

    for i,x in enumerate(backup_dir_list):
        if os.path.exists(dir_list[i]):
            shutil.copytree(dir_list[i], backup_dir_list[i])
            shutil.rmtree(dir_list[i])

def reset(dir_list):
    for i,x in enumerate(dir_list):
        if os.path.exists(dir_list[i]):
            shutil.rmtree(dir_list[i])

def makedir_if_nexist(dir_list):
    for cur_dir in dir_list:
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

def find_latest_checkpoint(filedir, sess_id):
    checkpoints = [os.path.join(filedir, x) for x in os.listdir(filedir) if '_'.join(x.split('-')[:-1]) == sess_id]
    if len(checkpoints) > 0:
        checkpoints.sort(key=os.path.getmtime)
        checkpoint_epochs = [int(os.path.splitext('_'.join(x.split('-')[1:]))[0]) for x in checkpoints]
        return checkpoints[-1], checkpoint_epochs[-1]
    else:
        return None, 0

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_optimal_thresholds(target, predictions):
    def get_opt_thresh(tprs, fprs, threshs):
        # Optimize at point where Sensitivity = Specificity
        opt_thresh = threshs[np.argmin(abs(tprs - (1 - fprs)))]
        return opt_thresh

    class_thresholds = []
    class_fprs = []
    class_tprs = []
    for iter_class in range(predictions.shape[1]):
        fprs, tprs, thresholds = roc_curve(target[:, iter_class], predictions[:, iter_class])
        opt_thresh = get_opt_thresh(tprs, fprs, thresholds)
        class_thresholds.append(opt_thresh)
        class_fprs.append(fprs)
        class_tprs.append(tprs)
    return class_thresholds, class_fprs, class_tprs


# Get thresholded class accuracies
def get_thresholded_metrics_class(target, predictions, thresholds):

    # Obtain thresholded predictions
    predictions_thresholded = predictions >= thresholds

    # Obtain metrics (class)
    cond_positive = np.sum(target == 1, 0)
    cond_negative = np.sum(target == 0, 0)

    true_positive = np.sum((target == 1) & (predictions_thresholded == 1), 0)
    false_positive = np.sum((target == 0) & (predictions_thresholded == 1), 0)
    true_negative = np.sum((target == 0) & (predictions_thresholded == 0), 0)
    false_negative = np.sum((target == 1) & (predictions_thresholded == 0), 0)

    class_tprs = true_positive / cond_positive
    class_fprs = false_positive / cond_negative
    class_tnrs = true_negative / cond_negative
    class_fnrs = false_negative / cond_positive

    class_accs = np.sum(target == predictions_thresholded, 0) / predictions_thresholded.shape[0]
    class_f1s = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)

    return class_tprs, class_fprs, class_tnrs, class_fnrs, class_accs, class_f1s

# Get thresholded overall accuracies
def get_thresholded_metrics_overall(target, predictions, thresholds):

    # Obtain thresholded predictions
    predictions_thresholded = predictions >= thresholds

    # Obtain metrics (class)
    cond_positive = np.sum(target == 1)
    cond_negative = np.sum(target == 0)

    true_positive = np.sum((target == 1) & (predictions_thresholded == 1))
    false_positive = np.sum((target == 0) & (predictions_thresholded == 1))
    true_negative = np.sum((target == 0) & (predictions_thresholded == 0))
    false_negative = np.sum((target == 1) & (predictions_thresholded == 0))

    overall_tpr = true_positive / cond_positive
    overall_fpr = false_positive / cond_negative
    overall_tnr = true_negative / cond_negative
    overall_fnr = false_negative / cond_positive

    overall_acc = np.sum(target == predictions_thresholded) / np.prod(predictions_thresholded.shape)
    overall_f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)

    return overall_tpr, overall_fpr, overall_tnr, overall_fnr, overall_acc, overall_f1

def plot_rocs(class_fprs, class_tprs, class_names, eval_dir, sess_id):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    for iter_class in range(len(class_names)):
        plt.plot(class_fprs[iter_class], class_tprs[iter_class], label=class_names[iter_class])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(eval_dir, 'ROC_' + sess_id + '.png'), bbox_inches='tight')
    plt.close()


def write_to_excel(metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s,
                   mean_tpr, mean_fpr, mean_tnr, mean_fnr, mean_acc, mean_f1, class_names, eval_dir, sess_id,
                   is_train=False):
    # Start a new Excel
    if not is_train:
        sess_xlsx_path = os.path.join(eval_dir, 'metrics_' + sess_id + '.xlsx')
    else:
        sess_xlsx_path = os.path.join(eval_dir, 'metrics_trainaug_' + sess_id + '.xlsx')
    df = pd.DataFrame({'HTT': class_names + ['Average'], 'TPR': list(metric_tprs) + [mean_tpr],
                       'FPR': list(metric_fprs) + [mean_fpr], 'TNR': list(metric_tnrs) + [mean_tnr],
                       'FNR': list(metric_fnrs) + [mean_fnr], 'ACC': list(metric_accs) + [mean_acc],
                       'F1': list(metric_f1s) + [mean_f1]}, columns=['HTT', 'TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'F1'])
    df.to_excel(sess_xlsx_path)

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

    # plt.xlabel('Prediction', fontsize=14)
    # axis_offset = -0.012*AUC.shape[0] + 1.436
    # ax.xaxis.set_label_coords(.5, axis_offset)

    fig.set_size_inches(cm2inch(fig_len, fig_len))