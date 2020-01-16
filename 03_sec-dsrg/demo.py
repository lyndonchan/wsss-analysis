import argparse
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', help='The WSSS method to be used (either SEC or DSRG)', type=str)
    parser.add_argument('-d', '--dataset', help='The dataset to run on (either ADP-morph, ADP-func, VOC2012, '
                                                'DeepGlobe_train75, or DeepGlobe_train37.5)', type=str)
    parser.add_argument('-n', '--setname', help='The name of the segmentation validation set in the ADP dataset, if '
                                                'applicable (either tuning or segtest)', type=str)
    parser.add_argument('-s', '--seed', help='The type of classification network to use for seeding (either VGG16, X1.7 for '
                                             'ADP-morph or ADP-func, or M7 for all other datasets)', type=str)
    parser.add_argument('-b', '--batchsize', help='The batch size', default=16, type=int)
    parser.add_argument('-i', '--saveimg', help='Toggle whether to save output segmentation as images', action='store_true')
    parser.add_argument('-v', '--verbose', help='Toggle verbosity of debug messages', action='store_true')

    args = parser.parse_args()

    assert(args.method and args.method in ['SEC', 'DSRG'])
    assert(args.dataset and args.dataset in ['ADP-morph', 'ADP-func', 'VOC2012', 'DeepGlobe_train75', 'DeepGlobe_train37.5'])
    assert(args.seed and args.seed in ['VGG16', 'X1.7', 'M7'])
    if args.dataset in ['ADP-morph', 'ADP-func']:
        assert(args.seed in ['VGG16', 'X1.7'])
        assert(args.setname and args.setname in ['tuning', 'segtest'])
    else:
        assert (args.seed in ['VGG16', 'M7'])
    assert(args.batchsize > 0)

    mdl = Model(args)
    mdl.load()
    mdl.predict()
