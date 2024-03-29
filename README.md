![logo](logo/logo.png)

# 🤖 farabio ❤️
[![PyPI version](https://img.shields.io/pypi/v/farabio)](https://badge.fury.io/py/farabio) [![DOI](https://zenodo.org/badge/314779309.svg)](https://zenodo.org/badge/latestdoi/314779309) ![PyPI - Downloads](https://img.shields.io/pypi/dm/farabio) [![Documentation Status](https://readthedocs.org/projects/farabio/badge/?version=latest)](https://farabio.readthedocs.io/en/latest) ![GitHub commit activity](https://img.shields.io/github/commit-activity/m/tuttelikz/farabio) 
[![GitHub](https://img.shields.io/github/license/tuttelikz/farabio)](https://opensource.org/licenses/Apache-2.0)

- [What's New](#-whats-new)
- [Introduction](#-introduction)
- [Features](#-features)
- [Biodatasets](#-biodatasets)
- [Models](#-models)
- [Getting Started (Installation)](#-getting-started-installation)
- [Tutorials](#-tutorials)
- [Links](#-links)
- [Credits](#-credits)
- [Licenses](#-licenses)
- [Acknowledgements](#-acknowledgements)

## 🎉 What's New

### August 26, 2021
Publishing `farabio==0.0.3` (*latest version*):  
[PyPI](https://pypi.org/project/farabio/0.0.3/) | [Release notes](https://github.com/tuttelikz/farabio/releases/tag/v0.0.3)

### August 18, 2021
Publishing `farabio==0.0.2`:  
[PyPI](https://pypi.org/project/farabio/0.0.2/) | [Release notes](https://github.com/tuttelikz/farabio/releases/tag/v0.0.2)

### April 21, 2021
This work is presented at PyTorch Ecosystem day. Poster is [here](https://pytorch.org/ecosystem/pted/2021).

### April 2, 2021
Publishing `farabio==0.0.1`:  
[PyPI](https://pypi.org/project/farabio/0.0.1/) | [Release notes](https://github.com/tuttelikz/farabio/releases/tag/v0.0.1)

### March 3, 2021
This work is selected for PyTorch Ecosystem Day.

## 💡 Introduction

**farabio** is a minimal PyTorch toolkit for out-of-the-box deep learning support in biomedical imaging. For further information, see [*Wikis*](https://github.com/tuttelikz/farabio/wiki) and [*Docs*](https://farabio.readthedocs.io).

## 🔥 Features
- Biomedical datasets
- Common DL models
- Flexible trainers (**in progress*)

## 📚 Biodatasets
* `biodatasets.ChestXrayDataset`:  
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
* `biodatasets.DSB18Dataset`:  
https://www.kaggle.com/c/data-science-bowl-2018/overview
* `biodatasets.HistocancerDataset`:  
https://www.kaggle.com/c/histopathologic-cancer-detection/overview
* `biodatasets.RANZCRDataset`:  
https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview
* `biodatasets.RetinopathyDataset`:  
https://www.kaggle.com/c/aptos2019-blindness-detection/overview
* `biodatasets.VinBigDataset`:  
 https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
* `biodatasets.MosmedDataset`:  
https://www.kaggle.com/datasets/andrewmvd/mosmed-covid19-ct-scans
* `biodatasets.EpiSeizureDataset`:
https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition

## 🚢 Models
### Classification:
* AlexNet - https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
* GoogLeNet - https://arxiv.org/pdf/1409.4842.pdf
* MobileNetV2 - https://arxiv.org/pdf/1801.04381.pdf
* MobileNetV3 - https://arxiv.org/pdf/1905.02244.pdf
* ResNet - https://arxiv.org/pdf/1512.03385.pdf
* ShuffleNetV2 - https://arxiv.org/pdf/1807.11164.pdf
* SqueezeNet - https://arxiv.org/pdf/1602.07360.pdf
* VGG - https://arxiv.org/pdf/1409.1556.pdf

### Segmentation:
* DeepLabV3 - https://arxiv.org/pdf/1706.05587
* U-Net - https://arxiv.org/pdf/1505.04597
* LinkNet - https://arxiv.org/pdf/1707.03718
* PSPNet - https://arxiv.org/pdf/1612.01105
* FPN - http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

### Volume:
* ConvNet3D - https://www.sciencedirect.com/science/article/pii/S0010482521001001

### Sequence:
* VanillaLSTM - https://www.sciencedirect.com/science/article/abs/pii/S001048251830132X

## 🚀 Getting started (Installation)

#### 1. Create and activate [`conda`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) environment:
```bash
conda create -n myenv python=3.8
conda activate myenv
```
#### 2. Install [`PyTorch`](https://pytorch.org/):
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3. Install [`farabio`](https://github.com/tuttelikz/farabio):
**A. With pip**:
```bash
pip install farabio
```

**B. Setup from source**:
```bash
git clone https://github.com/tuttelikz/farabio.git && cd farabio
pip install .
```

## 🤿 Tutorials
**Tutorial 1:** Training a classifier for `ChestXrayDataset` - [Notebook](https://github.com/tuttelikz/farabio/blob/main/farabio/notebooks/train-classifier.ipynb)  
**Tutorial 2:** Training a segmentation model for `DSB18Dataset` - [Notebook](https://github.com/tuttelikz/farabio/blob/main/farabio/notebooks/train-segmentation.ipynb)  
**Tutorial 3:** Training a Faster-RCNN detection model for `VinBigDataset` - [Notebook](https://github.com/tuttelikz/farabio/blob/main/farabio/notebooks/train-detection.ipynb)  
**Tutorial 4:** Training a 3D-CNN to predict the presence of viral pneumonia in computer tomography (CT) scans for `MosmedDataset` - [Script](https://github.com/tuttelikz/farabio/blob/main/farabio/notebooks/train-conv3d.py)  
**Tutorial 5:** Training a LSTM for epileptic seizures prediction using `EpiSeizureDataset` dataset - [Script](https://github.com/tuttelikz/farabio/blob/main/farabio/notebooks/train-lstm.py)


## 🔎 Links
- API documentations: https://farabio.readthedocs.io
- Code: https://github.com/tuttelikz/farabio
- Issue tracker: https://github.com/tuttelikz/farabio/issues
- PyPI package: https://pypi.org/project/farabio/
- Wiki: https://github.com/tuttelikz/farabio/wiki

## ⭐ Credits
If you like this repository, please click on Star.  

How to cite | [doi](https://zenodo.org/record/5746474#.YaclirqRX-g):
```text
@software{sanzhar_askaruly_2021_5746474,
  author       = {Sanzhar Askaruly and
                  Nurbolat Aimakov and
                  Alisher Iskakov and
                  Hyewon Cho and
                  Yujin Ahn and
                  Myeong Hoon Choi and
                  Hyunmo Yang and
                  Woonggyu Jung},
  title        = {Farabio: Deep learning for biomedical imaging},
  month        = dec,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.3-doi},
  doi          = {10.5281/zenodo.5746474},
  url          = {https://doi.org/10.5281/zenodo.5746474}
}
```
## 📃 Licenses
This work is licensed [Apache 2.0](https://github.com/tuttelikz/farabio/blob/main/LICENSE).

## 🤩 Acknowledgements
This work is based upon efforts of open-source PyTorch Community. I have tried to acknowledge related works (github links, arxiv papers) inside the source material, eg. README, documentation, and code docstrings. Please contact if I missed anything.
