
import torch
import torch.nn as nn
from net import common_cnn

class VGG(common_cnn.CommonCNN):

    def __init__(self, model_dir, tag, cfg, num_classes, use_cls, batchnorm=True):
        super(VGG, self).__init__(model_dir, tag, num_classes, use_cls, batchnorm)
        self.out_channels = 1024
        self.layer1 = common_cnn.make_layers(cfg[0], batchnorm, in_channels=3)
        self.layer2 = common_cnn.make_layers(cfg[1], batchnorm, in_channels=64)
        self.layer3 = common_cnn.make_layers(cfg[2], batchnorm, in_channels=128)
        self.layer4 = common_cnn.make_layers(cfg[3], batchnorm, in_channels=256)
        self.layer5 = common_cnn.make_layers(cfg[4], batchnorm, in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, self.num_classes),
            nn.Sigmoid()
        )
        self._load_pretrained(self.model_dir, self.tag)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg16_bn(model_dir, tag, num_classes, use_cls, batchnorm=True):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        model_dir (str): Directory holding the pre-trained CNN models, must contain 'models_CNN' at the end
        tag (str): Model name (containing dataset and CNN architecture)
        num_classes (int): Number of output classes
        batchnorm (bool): If True, use batch normalization layers
        pretrained (bool): If True, initializes model with pre-trained weights
    """
    cfg = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 512, 512, 512], [1024, 'D', 1024, 'D']]
    net = VGG(model_dir, tag, cfg, num_classes, use_cls, batchnorm=batchnorm)

    return net