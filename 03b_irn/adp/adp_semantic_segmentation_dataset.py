import numpy as np
import os
import cv2

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from chainercv.utils import read_label

CLS_COLOURS = {}
CLS_COLOURS['morph'] = np.array([(255, 255, 255), (0, 0, 128), (0, 128, 0), (255, 165, 0), (255, 192, 203), (255, 0, 0),
                                 (173, 20, 87), (176, 141, 105), (3, 155, 229), (158, 105, 175), (216, 27, 96),
                                 (244, 81, 30), (124, 179, 66), (142, 36, 255), (240, 147, 0), (204, 25, 165),
                                 (121, 85, 72), (142, 36, 170), (179, 157, 219), (121, 134, 203), (97, 97, 97),
                                 (167, 155, 142), (228, 196, 136), (213, 0, 0), (4, 58, 236), (0, 150, 136),
                                 (228, 196, 65), (239, 108, 0), (74, 21, 209)])
CLS_COLOURS['func'] = np.array([(255, 255, 255), (3, 155, 229), (0, 0, 128), (0, 128, 0), (173, 20, 87)])

class ADPSemanticSegmentationDataset(GetterDataset):

    """Semantic segmentation dataset for ADP."""

    def __init__(self, data_dir, htt_type, split='train'):
        super(ADPSemanticSegmentationDataset, self).__init__()
        self.htt_type = htt_type
        if self.htt_type not in ['morph', 'func']:
            raise ValueError('please pick HTT type from \'morph\', \'func\'')

        if split not in ['train', 'tuning', 'evaluation']:
            raise ValueError(
                'please pick split from \'train\', \'tuning\', \'evaluation\'')

        if split == 'evaluation':
            self.split = 'segtest'
        else:
            self.split = split
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(self.split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        if self.split == 'segtest':
            img_path = os.path.join(self.data_dir, 'PNGImagesSubset', self.ids[i] + '.png')
        else:
            img_path = os.path.join(self.data_dir, 'PNGImages', self.ids[i] + '.png')
        img = read_image(img_path, color=True)
        return img

    def _read_label(self, path, dtype=np.int32):
        gt_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
        label = np.zeros(gt_img.shape[:2], dtype=dtype)
        for iter_clr, clr in enumerate(CLS_COLOURS[self.htt_type]):
            gt_mask = np.all(gt_img == np.expand_dims(np.expand_dims(clr, 0), 0), axis=2)
            label += gt_mask * iter_clr
        return label

    def _get_label(self, i):
        label_path = os.path.join(
            self.data_dir, 'SegmentationClassAug', 'ADP-' + self.htt_type, self.ids[i] + '.png')
        label = self._read_label(label_path, dtype=np.int32)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label
