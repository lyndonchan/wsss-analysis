import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    figs_dir = 'figs'
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    models = ['SEC'] * 2
    datasets = ['ADP-morph'] * 2
    nets = ['VGG16', 'X1.7']
    ths = ['0.6'] * 2

    df_curr = {}

    for model, dataset, net, th in zip(models, datasets, nets, ths):
        idn = '%s_train_%s_%s' % (dataset, net, th)
        mdl_hist_dir = r'log/%s/%s' % (model, idn)
        files = sorted(os.listdir(mdl_hist_dir))

        mdl_hist_pth = os.path.join(mdl_hist_dir, files[-1])
        assert os.path.exists(mdl_hist_pth)

        # Read from tfevents
        hist = {}
        tags = ['epoch', 'lr', 'total_loss', 'miou_tuning', 'miou_segtest']
        series_names = ['epoch', 'lr']
        for tag in tags:
            hist[tag] = []
        for e in tf.train.summary_iterator(mdl_hist_pth):
            for v in e.summary.value:
                if v.tag in tags:
                    hist[v.tag].append(v.simple_value)
        df_curr[net] = pd.DataFrame(hist, columns=tags)

    # Plot training progress
    df_miou = pd.merge(df_curr[nets[0]].iloc[:, [0, 3, 4]], df_curr[nets[1]].iloc[:, [0, 3, 4]], on='epoch')
    df_miou.columns = ['Epoch', nets[0] + '-' + models[0] + ', val', nets[0] + '-' + models[0] + ', eval',
                       nets[1] + '-' + models[0] + ', val', nets[1] + '-' + models[0] + ', eval']
    df_loss = pd.merge(df_curr[nets[0]].iloc[:, [0, 2]], df_curr[nets[1]].iloc[:, [0, 2]], on='epoch')
    df_loss.columns = ['Epoch', nets[0] + '-' + models[0], nets[1] + '-' + models[0]]
    df_lr = pd.merge(df_curr[nets[0]].iloc[:, [0, 1]], df_curr[nets[1]].iloc[:, [0, 1]], on='epoch')
    df_lr.columns = ['Epoch', nets[0] + '-' + models[0], nets[1] + '-' + models[0]]

    ax = df_miou.plot(kind='line', x='Epoch', style=['C0', 'C0--', 'C1', 'C1--'])  # title=idn
    ax.set_ylabel('mIoU')
    ax = df_loss.plot(kind='line', x='Epoch')  # title=idn
    ax.set_ylabel('Training Loss')
    ax = df_lr.plot(kind='line', x='Epoch')  # title=idn
    ax.set_ylabel('Learning Rate')
    plt.show()
    a=1