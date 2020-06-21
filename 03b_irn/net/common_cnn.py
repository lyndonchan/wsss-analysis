import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import tensorflow as tf
import keras
from keras import backend as K
import scipy.io as io

from keras.models import model_from_json
import numpy as np

class CommonCNN(nn.Module):
    def __init__(self, model_dir, tag, num_classes, use_cls, batchnorm=True):
        super(CommonCNN, self).__init__()
        self.model_dir = model_dir
        self.tag = tag
        self.num_classes = num_classes
        self.use_cls = use_cls
        self.batchnorm = batchnorm
        self.thresholds = torch.from_numpy(.5 * np.ones(self.num_classes, dtype=np.float32))

    def _load_pretrained(self, model_dir, tag, gen_gradcam=False):
        model_path = os.path.join(model_dir, tag, tag + '.json')
        weights_path = os.path.join(model_dir, tag, tag + '.h5')
        thresh_path = os.path.join(model_dir, tag, tag + '.mat')
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
        with tf.Session(config=config) as sess:
            K.set_session(sess)
            model = model_from_json(loaded_model_json)
            model.load_weights(weights_path)
            weights = model.get_weights()
            self.load_weights_from_file(weights)
            thresholds = np.maximum(io.loadmat(thresh_path).get('optimalScoreThresh')[0], 1/3)
            self.load_thresholds_from_file(thresholds)
            if gen_gradcam:
                self.gen_gradcam_weights(model)

    def load_weights_from_file(self, weights):
        def HWCD_to_DCHW(w):
            return np.transpose(w, (3, 2, 0, 1))
        use_bias = 'VGG16' not in self.tag
        conv2d_layers = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        batchnorm2d_layers = [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        assert 2*len(conv2d_layers) + 4*len(batchnorm2d_layers) + (1+use_bias)*len(linear_layers) == len(weights), \
               'Sizes of PyTorch network and saved Keras network differ!'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                w = HWCD_to_DCHW(weights.pop(0))
                assert tuple(m.weight.data.shape) == tuple(w.shape)
                m.weight.data = torch.from_numpy(w)
                w = weights.pop(0)
                assert tuple(m.bias.data.shape) == tuple(w.shape)
                m.bias.data = torch.from_numpy(w)
            elif isinstance(m, nn.BatchNorm2d):
                w = weights.pop(0)
                assert tuple(m.weight.data.shape) == tuple(w.shape)
                m.weight.data = torch.from_numpy(w)
                w = weights.pop(0)
                assert tuple(m.bias.data.shape) == tuple(w.shape)
                m.bias.data = torch.from_numpy(w)
                w = weights.pop(0)
                assert tuple(m.running_mean.data.shape) == tuple(w.shape)
                m.running_mean.data = torch.from_numpy(w)
                w = weights.pop(0)
                assert tuple(m.running_var.data.shape) == tuple(w.shape)
                m.running_var.data = torch.from_numpy(w)
            elif isinstance(m, nn.Linear):
                w = np.transpose(weights.pop(0))
                assert m.weight.data.shape[0] <= w.shape[0]
                m.weight.data = torch.from_numpy(w)
                if use_bias:
                    w = weights.pop(0)
                    assert m.bias.data.shape[0] <= w.shape[0]
                    m.bias.data = torch.from_numpy(w)

    def gen_gradcam_weights(self, input_model, should_normalize=True):
        """Obtain Grad-CAM weights of the model

        Parameters
        ----------
        input_model : keras.engine.sequential.Sequential object
            The input model
        should_normalize : bool, optional
            Whether to normalize the gradients
        """

        def find_final_layer(model):
            for iter_layer, layer in reversed(list(enumerate(model.layers))):
                if type(layer) == type(layer) == keras.layers.convolutional.Conv2D:
                    return model.layers[iter_layer + 1].name
            raise Exception('Could not find the final layer in provided HistoNet')

        final_layer = find_final_layer(input_model)
        conv_output = input_model.get_layer(final_layer).output  # activation_7
        num_classes = input_model.output_shape[1]
        num_feats = int(conv_output.shape[-1])
        self.gradcam_weights = np.zeros((num_feats, num_classes))

        def normalize(x):
            # utility function to normalize a tensor by its L2 norm
            return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

        dummy_image = np.zeros([1] + list(input_model.input_shape[1:]), dtype=np.float32)
        for iter_class in range(input_model.output_shape[1]):
            y_c = input_model.layers[-2].output[0, iter_class]
            if should_normalize:
                grad = normalize(K.gradients(y_c, conv_output)[0])
            else:
                grad = K.gradients(y_c, conv_output)[0]
            grad_func = K.function([input_model.layers[0].input, K.learning_phase()], [conv_output, grad])
            conv_val, grad_val = grad_func([dummy_image, 0])
            conv_val, grad_val = conv_val[0], grad_val[0]
            self.gradcam_weights[:, iter_class] = np.mean(grad_val, axis=(0, 1))

    def load_thresholds_from_file(self, thresholds):
        assert self.num_classes <= len(thresholds)
        self.thresholds = torch.from_numpy(np.float32(thresholds))


def make_layers(layer, batchnorm, in_channels=3):
    layers = []
    for v in layer:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'D':
            layers += [nn.Dropout(p=0.5)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batchnorm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v, eps=0.001, momentum=0.99)] # reversed
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)