# farabio - Deep learning toolkit for biomedical imaging

![logo](logo/Final_Cropped_3.png)

[![GitHub](https://img.shields.io/github/license/tuttelikz/farabio)](https://opensource.org/licenses/MIT) [![Documentation Status](https://readthedocs.org/projects/farabio/badge/?version=latest)](https://farabio.readthedocs.io/en/latest)
[![PyPI version](https://img.shields.io/pypi/v/farabio)](https://badge.fury.io/py/farabio)

**farabio** is a minimal [open-source](LICENSE) Python package based on [PyTorch](https://pytorch.org/) for out-of-the-box deep learning support in biomedical imaging. 

## Features

- Image (pre)processing
- Flexible PyTorch trainers
- Common DL models
- Biomedical datasets
- Loggers, visualization 

## Getting started

### 1. Activate conda environment:

```bash
$ conda create -n coolenv python=3.8
$ conda activate coolenv
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

### 2. Install prerequisites:

```bash
$ git clone https://github.com/tuttelikz/farabio.git && cd farabio
$ pip install -r requirements.txt
```

### 3. Install **farabio**:

**A. With pip**:
```bash
$ pip install farabio 
```

**B. Setup from source**:
```bash
$ pip install [-e] .      # flag for editable mode
```

## Updates
[21-04-02] Release of [`farabio==0.0.1`](https://pypi.org/project/farabio/)  
[21-03-27] This work is selected for poster on [PyTorch Ecosystem Day](https://pytorchecosystemday.fbreg.com/) to be held on April 21, 2021.

## Team of contributors

San Askaruly [@tuttelikz](https://github.com/tuttelikz)  
Nurbolat Aimakov [@aimakov](https://github.com/aimakov)  
Alisher Iskakov [@finesome](https://github.com/finesome)

## Links

- API documentations: https://farabio.readthedocs.io
- Code: https://github.com/tuttelikz/farabio
- Issue tracker: https://github.com/tuttelikz/farabio/issues
- PyPI package: https://pypi.org/project/farabio/
- Wiki: https://github.com/tuttelikz/farabio/wiki

## Acknowledgement

This work started as a fun project at Translational Biophotonics Lab, UNIST, and was supported with computational resources of the laboratory.
