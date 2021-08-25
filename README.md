![logo](logo/logo.png)

[![GitHub](https://img.shields.io/github/license/tuttelikz/farabio)](https://opensource.org/licenses/Apache-2.0) [![Documentation Status](https://readthedocs.org/projects/farabio/badge/?version=latest)](https://farabio.readthedocs.io/en/latest)
[![PyPI version](https://img.shields.io/pypi/v/farabio)](https://badge.fury.io/py/farabio)

# farabio
- [What's New](#whats-new)
- [Introduction](#introduction)
- [Features](#features)
- [Biodatasets](#biodatasets)
- [Models](#models)
- [Getting Started (Installation)](#getting-started-installation)
- [Tutorials](#tutorials)
- [Links](#links)
- [Credits](#credits)
- [Licenses](#licenses)
- [Acknowledgements](#acknowledgements)

## What's New

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

## Introduction

**farabio** is a minimal PyTorch toolkit for out-of-the-box deep learning support in biomedical imaging. For further information, see [*Wikis*](https://github.com/tuttelikz/farabio/wiki) and [*Docs*](https://farabio.readthedocs.io).

## Features
- Biomedical datasets
- Common DL models
- Flexible trainers (**in progress*)

## Biodatasets
* Chest X-Ray Images (Pneumonia):  
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
* 2018 Data Science Bowl:  
https://www.kaggle.com/c/data-science-bowl-2018/overview
* Histopathologic Cancer Detection:  
https://www.kaggle.com/c/histopathologic-cancer-detection/overview
* RANZCR CLiP - Catheter and Line Position Challenge:  
https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview
* APTOS 2019 Blindness Detection:  
https://www.kaggle.com/c/aptos2019-blindness-detection/overview

## Models
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

## Getting started (Installation)

#### 1. Activate [`conda`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) environment:
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

## Tutorials
**Tutorial 1:** Training a classifier for `ChestXrayDataset` - [Notebook](https://github.com/tuttelikz/farabio/blob/main/farabio/notebooks/train-classifier.ipynb)  
**Tutorial 2:** Training a segmentation model for `DSB18Dataset` - [Notebook](https://github.com/tuttelikz/farabio/blob/main/farabio/notebooks/train-segmentation.ipynb)

## Links
- API documentations: https://farabio.readthedocs.io
- Code: https://github.com/tuttelikz/farabio
- Issue tracker: https://github.com/tuttelikz/farabio/issues
- PyPI package: https://pypi.org/project/farabio/
- Wiki: https://github.com/tuttelikz/farabio/wiki

## Credits
If you like this repository, please click on Star.

## Licenses
This work is licensed [Apache 2.0](https://github.com/tuttelikz/farabio/blob/main/LICENSE).

## Acknowledgements
This work is based upon efforts of open-source PyTorch Community. I have tried to acknowledge related works (github links, arxiv papers) inside the source material, eg. README, documentation, and code docstrings. Please contact if I missed anything.
