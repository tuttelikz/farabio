![logo](logo/logo.png)

[![GitHub](https://img.shields.io/github/license/tuttelikz/farabio)](https://opensource.org/licenses/Apache-2.0) [![Documentation Status](https://readthedocs.org/projects/farabio/badge/?version=latest)](https://farabio.readthedocs.io/en/latest)
[![PyPI version](https://img.shields.io/pypi/v/farabio)](https://badge.fury.io/py/farabio)

# farabio
- [What's New](#whats-new)
- [Introduction](#introduction)
- [Features](#features)
- [Biodatasets](#biodatasets)
- [Models](#models)
- [Getting Started (Installation)](#getting-started)
- [Licenses](#licenses)
- [Links](#links)

## What's New

### August 17, 2021
Release of [`farabio==0.0.2`](https://pypi.org/project/farabio/)

### April 2, 2021
Release of [`farabio==0.0.1`](https://pypi.org/project/farabio/)  

### March 3, 2021
[21-03-27] This work is presented on PyTorch Ecosystem Day (21/04/21). Poster is [here](https://pytorch.org/ecosystem/pted/2021)

## Introduction

**farabio** is a minimal [PyTorch](https://pytorch.org/) toolkit for out-of-the-box deep learning in biomedical imaging.

## Features
- Biomedical datasets
- Common DL models
- Flexible trainers (*in progress)

## Biodatasets
* Chest X-Ray Images (Pneumonia) - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
* 2018 Data Science Bowl - https://www.kaggle.com/c/data-science-bowl-2018/overview
* Histopathologic Cancer Detection - https://www.kaggle.com/c/histopathologic-cancer-detection/overview
* RANZCR CLiP - Catheter and Line Position Challenge - https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview
* APTOS 2019 Blindness Detection - https://www.kaggle.com/c/aptos2019-blindness-detection/overview

## Models
* AlexNet - https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
* GoogLeNet - https://arxiv.org/pdf/1409.4842.pdf
* MobileNetV2 - https://arxiv.org/pdf/1801.04381.pdf
* MobileNetV3 - https://arxiv.org/pdf/1905.02244.pdf
* ResNet - https://arxiv.org/pdf/1512.03385.pdf
* ShuffleNetV2 - https://arxiv.org/pdf/1807.11164.pdf
* SqueezeNet - https://arxiv.org/pdf/1602.07360.pdf
* VGG - https://arxiv.org/pdf/1409.1556.pdf

## Getting started (Installation)

### 1. Activate conda environment:
```bash
$ conda create -n myenv python=3.8
$ conda activate myenv
```

### 2. Install **farabio**:
**A. With pip**:
```bash
$ pip install farabio -f https://download.pytorch.org/whl/torch_stable.html
```

**B. Setup from source**:
```bash
# [-e] is flag for editable mode
$ pip install [-e] . -f https://download.pytorch.org/whl/torch_stable.html
```

## Licenses


## Links

- API documentations: https://farabio.readthedocs.io
- Code: https://github.com/tuttelikz/farabio
- Issue tracker: https://github.com/tuttelikz/farabio/issues
- PyPI package: https://pypi.org/project/farabio/
- Wiki: https://github.com/tuttelikz/farabio/wiki
