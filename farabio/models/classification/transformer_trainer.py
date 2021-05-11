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


# training parameters
batch_size = 64
epochs = 5
lr = 3e-5
gamma = 0.7
seed = 42
device = "cuda"

root = "/home/data/02_SSD4TB/suzy/datasets/public"

# Defining a function to seed everything.
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Running the function:
seed_everything(seed)

train_dataset = RANZCRDataset(
    root=root, train=True, transform=None, download=False)
valid_dataset = RANZCRDataset(
    root=root, train=False, transform=None, download=False)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=batch_size, shuffle=True)

# defining model
efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=11,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# Defining some other presets:
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# start traububg kiios
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    i = 0
    for data, label in tqdm(train_loader):
        print(i)
        i = i+1
        data = data.to(device)
        label = label.to(device)

        output = model(data)

        #label = torch.nn.functional.one_hot(label, num_classes=11)
        label = label.type_as(output)

        #print(label.dtype)
        #print(output.dtype)
        label = label.type(torch.cuda.LongTensor)
        #label = label.to(device)

        #print(label.device.index)
        #print(output.device.index)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    j = 0
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            print(j)
            j = j+1
            data = data.to(device)
            label = label.to(device)
            
            val_output = model(data)
            
            #label = torch.cuda.LongTensor(label)
            #label = label.type(torch.cuda.LongTensor)
            label = label.type(torch.cuda.LongTensor)
            #label = label.to(device)

            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch: {epoch+1} - loss: {epoch_loss:.4f} - acc : {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy: .4f}\n"
    )
