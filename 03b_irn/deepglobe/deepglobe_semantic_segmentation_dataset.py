import numpy as np
import os
import cv2

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from chainercv.utils import read_label

CLS_COLOURS = np.array([(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255),
                        (255, 255, 255)])

class DeepGlobeSemanticSegmentationDataset(GetterDataset):

    """Semantic segmentation dataset for ADP."""

    def __init__(self, data_dir, is_balanced=False, split='train'):
        super(DeepGlobeSemanticSegmentationDataset, self).__init__()
        self.is_balanced = is_balanced

        if split not in ['train', 'test']:
            raise ValueError(
                'please pick split from \'train\', \'test\'')

        if split == 'train' and not self.is_balanced:
            self.split = 'train75'
        elif split == 'train' and self.is_balanced:
            self.split = 'train37.5'
        elif split == 'test':
            self.split = 'test'
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(self.split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = read_image(img_path, color=True)
        return img

    def _read_label(self, path, dtype=np.int32):
        gt_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
        label = np.zeros(gt_img.shape[:2], dtype=dtype)
        for iter_clr, clr in enumerate(CLS_COLOURS):
            gt_mask = np.all(gt_img == np.expand_dims(np.expand_dims(clr, 0), 0), axis=2)
            label += gt_mask * iter_clr
        return label

    def _get_label(self, i):
        label_path = os.path.join(
            self.data_dir, 'SegmentationClassAug', self.ids[i] + '.png')
        label = self._read_label(label_path, dtype=np.int32)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label
