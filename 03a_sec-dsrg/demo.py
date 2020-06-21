import argparse
from model import Model

def _run(cmd):
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--task', help='The task to run (either train or predict)', type=str)
    parser.add_argument('-m', '--method', help='The WSSS method to be used (either SEC or DSRG)', type=str)
    parser.add_argument('-d', '--dataset', help='The dataset to run on (either ADP-morph, ADP-func, VOC2012, '
                                                'DeepGlobe, or DeepGlobe_balanced)', type=str)
    parser.add_argument('-n', '--eval_setname', help='The name of the segmentation validation set in the ADP dataset, '
                                                     'if applicable (either tuning or segtest)', type=str)
    parser.add_argument('-s', '--seed', help='The type of classification network to use for seeding (either VGG16, X1.7 '
                                             'for ADP-morph or ADP-func, or M7 for all other datasets)', type=str)
    parser.add_argument('-t', '--threshold', help='The threshold level for discretizing activation maps', default=0.2,
                        type=float)
    parser.add_argument('-b', '--batchsize', help='The batch size', default=16, type=int)
    parser.add_argument('-e', '--epochs', help='The number of epochs to train for', type=int)
    parser.add_argument('-i', '--saveimg', help='Toggle whether to save output segmentation as images', action='store_true')
    parser.add_argument('-v', '--verbose', help='Toggle verbosity of debug messages', action='store_true')

    args = parser.parse_args(cmd)
    print(args)

    assert(args.task and args.task in ['train', 'predict'])
    assert(args.method and args.method in ['SEC', 'DSRG'])
    assert(args.dataset and args.dataset in ['ADP-morph', 'ADP-func', 'VOC2012', 'DeepGlobe', 'DeepGlobe_balanced'])
    assert(args.seed and args.seed in ['VGG16', 'X1.7', 'M7'])
    if args.dataset in ['ADP-morph', 'ADP-func']:
        assert(args.seed in ['VGG16', 'X1.7'])
    else:
        assert (args.seed in ['VGG16', 'M7'])
    assert(args.batchsize > 0)
    if args.task == 'train':
        assert(args.epochs > 0)

    mdl = Model(args)
    mdl.load()

    if args.task == 'train':
        mdl.train()
    elif args.task == 'predict':
        if args.dataset in ['ADP-morph', 'ADP-func']:
            assert (args.eval_setname and args.eval_setname in ['tuning', 'segtest'])
        mdl.predict()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        _run(sys.argv[1:])
    else:
        cfg = []
        for dataset in ['DeepGlobe_balanced']: # ['VOC2012']:
            for method in ['SEC', 'DSRG']:
                for net in ['VGG16', 'X1.7']:
                    if dataset in ['ADP-morph', 'ADP-func']:
                        th = '0.9'
                        epochs = '8'
                    elif dataset == 'VOC2012':
                        if net == 'X1.7':
                            net = 'M7'
                        th = '0.2'
                        if method == 'SEC':
                            epochs = '16'
                        elif method == 'DSRG':
                            epochs = '6'
                    elif 'DeepGlobe' in dataset:
                        if net == 'X1.7':
                            net = 'M7'
                        if not (dataset == 'DeepGlobe' and net == 'M7'):
                            th = '0.3'
                        else:
                            th = '0.4'
                        epochs = '100'
                    cfg.append(['--task', 'train', '--method', method, '--dataset', dataset, '--seed', net,
                                '--threshold', th, '--epochs', epochs, '-v'])
                    if dataset in ['ADP-morph', 'ADP-func']:
                        cfg.append(['--task', 'predict', '--method', method, '--dataset', dataset,
                                    '--eval_setname', 'tuning', '--seed', net, '--threshold', th, '-v'])
                        cfg.append(['--task', 'predict', '--method', method, '--dataset', dataset,
                                    '--eval_setname', 'segtest', '--seed', net, '--threshold', th, '-v'])
                    else:
                        cfg.append(['--task', 'predict', '--method', method, '--dataset', dataset,
                                    '--seed', net, '--threshold', th, '-v'])
        for c in cfg:
            _run(c)
