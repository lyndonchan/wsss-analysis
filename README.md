# wsss-analysis

The code of: A Comprehensive Analysis of Weakly-Supervised Semantic Segmentation in Different Image Domains, arXiv pre-print 2019 [paper](https://arxiv.org/abs/1912.11186).



## Introduction
![](/methods.png)

We conduct the first comprehensive analysis of Weakly-Supervised Semantic Segmentation (WSSS) with image label supervision in different image domains. WSSS has been almost exclusively evaluated on PASCAL VOC2012 but little work has been done on applying to different image domains, such as histopathology and satellite images. The paper analyzes the compatibility of different methods for representative datasets and presents principles for applying to an unseen dataset.

![](/method.png)

In this repository, we provide the evaluation code used to generate the weak localization cues and final segmentations from Section 5 (Performance Evaluation) of the paper. The code release enables reproducing the results in our paper. The Keras implementation of HistoSegNet was adapted from [hsn_v1](https://github.com/lyndonchan/hsn_v1); the Tensorflow implementations of SEC and DSRG were adapted from [SEC-tensorflow](https://github.com/xtudbxk/SEC-tensorflow) and [DSRG-tensorflow](https://github.com/xtudbxk/DSRG-tensorflow), respectively. The PyTorch implementation of IRNet was adapted from [irn](https://github.com/jiwoon-ahn/irn). Pretrained models and evaluation images are also available for download.

## Citing this repository

If you find this code useful in your research, please consider citing us:

        @misc{chan2019comprehensive,
            title={A Comprehensive Analysis of Weakly-Supervised Semantic Segmentation in Different Image Domains},
            author={Lyndon Chan and Mahdi S. Hosseini and Konstantinos N. Plataniotis},
            year={2019},
            eprint={1912.11186},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

Mandatory

* `python` (checked on 3.5)
* `scipy` (checked on 1.2.0)
* `skimage` / `scikit-image` (checked on 0.15.0)
* `keras` (checked on 2.2.4)
* `tensorflow` (checked on 1.13.1)
* `tensorflow-gpu` (checked on 1.13.1)
* `numpy` (checked on 1.18.1)
* `pandas` (checked on 0.23.4)
* `cv2` / `opencv-python` (checked on 3.4.4.19)
* `cython`
* `imageio` (checked on 2.5.0)
* `chainercv` (checked on 0.12.0)
* `pydensecrf` (git+https://github.com/lucasb-eyer/pydensecrf.git)
* `torch` (checked on 1.1.0)
* `torchvision` (checked on 0.2.2.post3)
* `tqdm`

Optional

* `matplotlib` (checked on 3.0.2)
* `jupyter`

To utilize the code efficiently, GPU support is required. The following configurations have been tested to work successfully:
* CUDA Version: 10
* CUDA Driver Version: r440
* CUDNN Version: 7.6.4 - 7.6.5
We do not guarantee proper functioning of the code using different versions of CUDA or CUDNN.

## Hardware Requirements
Each method used in this repository has different GPU memory requirements. We have listed the approximate GPU memory requirements for each model through our own experiments:
* `01_train`: ~6 GB (e.g. NVIDIA RTX 2060)
* `02_cues`: ~6 GB (e.g. NVIDIA RTX 2060)
* `03a_sec-dsrg`: ~11 GB (e.g. NVIDIA GTX 2080 Ti)
* `03b_irn`: ~8 GB (e.g. NVIDIA GTX 1070)
* `03c_hsn`: ~6 GB (e.g. NVIDIA RTX 2060)

## Downloading data

The pretrained models, ground-truth annotations, and images used in this paper are available on Zenodo under a Creative Commons Attribution license: [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.3902506.svg)](http://dx.doi.org/10.5281/zenodo.3902506). Please extract the contents into your `wsss-analysis\database` directory. If you choose to extract the data to another directory, please modify the filepaths accordingly in `settings.ini`.

Note: the training-set images of ADP are released on a case-by-case basis due to the confidentiality agreement for releasing the data. To obtain access to `wsss-analysis\database\ADPdevkit\ADPRelease1\JPEGImages` and `wsss-analysis\database\ADPdevkit\ADPRelease1\PNGImages` needed for `gen_cues` in `01_weak_cues`, apply for access separately [here](http://www.dsp.utoronto.ca/projects/ADP/ADP_Database/).

## Running the code

### Scripts
To run `02_cues` (generate weak cues for SEC and DSRG):
```
cd 02_cues
python demo.py
```

To run `03a_sec-dsrg` (train/evaluate SEC, DSRG performance in Section 5; to omit training, comment out lines 76-77 in `03a_sec-dsrg\demo.py`):
```
cd 03a_sec-dsrg
python demo.py
```

To run `03b_irn` (train/evaluate IRNet and Grad-CAM performance in Section 5):
```
cd 03b_irn
python demo_tune.py
```

To run `03b_irn` (evaluate pre-trained Grad-CAM performance in Section 5):
```
cd 03b_irn
python demo_cam.py
```

To run `03b_irn` (evaluate pre-trained IRNet performance in Section 5):
```
cd 03b_irn
python demo_sem_seg.py
```

To run `03c_hsn` (evaluate HistoSegNet performance in Section 5):
```
cd 03c_hsn
python demo.py
```

### Notebooks

`03a_sec-dsrg`:
* VGG16-SEC on ADP-morph: [03a_sec-dsrg/03a_sec-adp-morph.ipynb](03a_sec-dsrg/03a_sec-adp-morph.ipynb)
* VGG16-SEC on ADP-func: [03a_sec-dsrg/03a_sec-adp-func.ipynb](03a_sec-dsrg/03a_sec-adp-func.ipynb)
* VGG16-SEC on VOC2012: [03a_sec-dsrg/03a_sec-voc2012.ipynb](03a_sec-dsrg/03a_sec-voc2012.ipynb)
* VGG16-SEC on DeepGlobe: [03a_sec-dsrg/03a_sec-deepglobe.ipynb](03a_sec-dsrg/03a_sec-deepglobe.ipynb)

`03b_irn`:
* VGG16-IRNet on ADP-morph: (TODO)
* VGG16-IRNet on ADP-func: (TODO)
* VGG16-IRNet on VOC2012: (TODO)
* VGG16-IRNet on DeepGlobe: (TODO)

`03c_hsn`:
* VGG16-HistoSegNet on ADP: [03c_hsn/03c_hsn-adp.ipynb](03c_hsn/03c_hsn-adp.ipynb)
* VGG16-HistoSegNet on VOC2012: [03c_hsn/03c_hsn-voc2012.ipynb](03c_hsn/03c_hsn-voc2012.ipynb)
* VGG16-HistoSegNet on DeepGlobe: [03c_hsn/03c_hsn-deepglobe.ipynb](03c_hsn/03c_hsn-deepglobe.ipynb)


## Results
To access each method's evaluation results, check the associated `eval` (for numerical results) and `out` (for outputted images) folders. For easy access to all evaluated results, run `scripts/extract_eval.py`.

(NOTE: the numerical results obtained for SEC and DSRG DeepGlobe_balanced differ slightly from those reported in the paper due to retraining the models during code cleanup. Also, `tuning` is equivalent to the validation set and `segtest` is equivalent to the evaluation set in ADP. See [hsn_v1](https://github.com/lyndonchan/hsn_v1) to replicate those results for ADP precisely.)

| Network | -         | -       | VGG16    | -        | -        | -           | -           | X1.7/M7  | -        | -        | -           | -           |
|-------------------|-----------|---------|----------|----------|----------|-------------|-------------|----------|----------|----------|-------------|-------------|
| WSSS Method | -         | -       | Grad-CAM | SEC      | DSRG     |IRNet| HistoSegNet | Grad-CAM | SEC      | DSRG     |IRNet| HistoSegNet |
| Dataset   | Training  | Testing |    "      |     "     |   "       |     "     |     "     |     "     |     "     |     "        |     "     | " |
| ADP-morph | train                 | validation |0.14507|0.10730|0.08826|0.15068|0.13255|0.20997|0.13597|0.13458|0.21450|0.27546|
| ADP-morph | train                 | evaluation |0.14946|0.11409|0.08011|0.15546|0.16159|0.21426|0.13369|0.10835|0.21737|0.26156|
| ADP-func  | train                 | validation |0.34813|0.28232|0.37193|0.35016|0.44215|0.35233|0.32216|0.28625|0.34730|0.50663|
| ADP-func  | train                 | evaluation |0.38187|0.28097|0.44726|0.36318|0.44115|0.37910|0.30828|0.31734|0.38943|0.48020|
| VOC2012   | train                 | val        |0.26262|0.37058|0.32129|0.31198|0.22707|0.14946|0.37629|0.35004|0.17844|0.09201|
| DeepGlobe | training (75% test)   | evaluation (25% test) |0.28037|0.24005|0.28841|0.29405|0.24019|0.21260|0.24841|0.35258|0.24620|0.29398|
| DeepGlobe | training (37.5% test) | evaluation (25% test) |0.28083|0.25512|0.32017|0.29207|0.30410|0.22266|0.20050|0.26470|0.21303|0.21617|


## Examples

### ADP-morph
![](/qual_eval_ADP-morph.png)

### ADP-func
![](/qual_eval_ADP-func.png)

### VOC2012
![](/qual_eval_VOC2012.png)

### DeepGlobe
![](/qual_eval_DeepGlobe.png)

## TODO
1. Improve comments and code documentation
2. Add IRNet notebooks
3. Clean up IRNet code