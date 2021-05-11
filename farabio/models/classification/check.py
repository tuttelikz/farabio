import glob
from itertools import chain
import os
import random
import zipfile

# Our all season best friends:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch because survival:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# To keep time in check:
from tqdm.notebook import tqdm

# For Preprocessing of the Data:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# For grabbing our pretrained Model:
from farabio.data.biodatasets import RANZCRDataset
from farabio.models.classification.vit.linformer import Linformer
from farabio.models.classification.vit.efficient import ViT


root = "/home/data/02_SSD4TB/suzy/datasets/public"

train_dataset = RANZCRDataset(
    root=root, train=True, transform=None, download=False)
valid_dataset = RANZCRDataset(
    root=root, train=False, transform=None, download=False)

batch_size = 64
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = True)

print(len(train_loader))
print(len(valid_loader))