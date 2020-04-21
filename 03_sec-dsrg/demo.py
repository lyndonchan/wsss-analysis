import argparse
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--task', help='The task to run (either train or predict)', type=str)
    parser.add_argument('-m', '--method', help='The WSSS method to be used (either SEC or DSRG)', type=str)
    parser.add_argument('-d', '--dataset', help='The dataset to run on (either ADP-morph, ADP-func, VOC2012, '
                                                'DeepGlobe_train75, or DeepGlobe_train37.5)', type=str)
    parser.add_argument('-n', '--eval_setname', help='The name of the segmentation validation set in the ADP dataset, '
                                                     'if applicable (either tuning or segtest)', type=str)
    parser.add_argument('-s', '--seed', help='The type of classification network to use for seeding (either VGG16, X1.7 '
                                             'for ADP-morph or ADP-func, or M7 for all other datasets)', type=str)
    parser.add_argument('-t', '--threshold', help='The threshold level for discretizing activation maps', default=0.2,
                        type=float)
    parser.add_argument('-b', '--batchsize', help='The batch size', default=16, type=int)
    parser.add_argument('-e', '--epochs', help='The number of epochs to train for', type=int) # {'ADP': 8, 'VOC2012': 16, 'DeepGlobe', 13}
    parser.add_argument('-i', '--saveimg', help='Toggle whether to save output segmentation as images', action='store_true')
    parser.add_argument('-v', '--verbose', help='Toggle verbosity of debug messages', action='store_true')

    args = parser.parse_args()

    assert(args.task and args.task in ['train', 'predict'])
    assert(args.method and args.method in ['SEC', 'DSRG'])
    assert(args.dataset and args.dataset in ['ADP-morph', 'ADP-func', 'VOC2012', 'DeepGlobe_train75', 'DeepGlobe_train37.5'])
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