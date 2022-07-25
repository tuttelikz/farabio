import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile
import io
import wget
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F


class MosmedDataset(Dataset):
    def __init__(self, download: bool = False, save_path: str = ".", train: bool = True):
        if download:
            ct_link = {
                "CT-0": "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip",
                "CT-23": "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
            }
            for key in ct_link.keys():
                wget.download(ct_link[key], save_path)
                with zipfile.ZipFile(os.path.join(save_path, ct_link + ".zip"), "r") as z_fp:
                    z_fp.extractall(os.path.join(save_path))

        volume_dir = "/home/data/01_SSD4TB/suzy/datasets/public-datasets/mosmed"
        normal_scans = [os.path.join(volume_dir, "CT-0", fname)
                        for fname in os.listdir(os.path.join(volume_dir, "CT-0"))]
        abnormal_scans = [os.path.join(volume_dir, "CT-23", fname)
                          for fname in os.listdir(os.path.join(volume_dir, "CT-23"))]
        normal_labels = [[0] for _ in range(len(normal_scans))]
        abnormal_labels = [[1] for _ in range(len(abnormal_scans))]

        normal = list(zip(normal_scans, normal_labels))
        abnormal = list(zip(abnormal_scans, abnormal_labels))

        all_list = normal + abnormal
        train_files, test_files = train_test_split(all_list, test_size=0.3)
        if train:
            self.fnames = train_files
        else:
            self.fnames = test_files

    def normalize_volume(self, volume):
        _min, _max = -1000, 400
        volume[volume < _min] = _min
        volume[volume > _max] = _max
        volume = (volume - _min) / (_max - _min)
        volume = volume.astype("float32")
        return volume

    def resize_volume(self, volume):
        target_d, target_w, target_h = 64, 128, 128
        curr_d, curr_w, curr_h = volume.shape[-1], volume.shape[0], volume.shape[1]

        d = curr_d / target_d
        w = curr_w / target_w
        h = curr_h / target_h

        d_factor = 1 / d
        w_factor = 1 / w
        h_factor = 1 / h

        volume = ndimage.rotate(volume, 90, reshape=False)
        volume = ndimage.zoom(volume, (w_factor, h_factor, d_factor), order=1)

        return volume

    def __getitem__(self, idx: int):
        volume = nib.load(self.fnames[idx][0])
        label = self.fnames[idx][-1][0]
        volume = volume.get_fdata()
        volume = self.normalize_volume(volume)
        volume = self.resize_volume(volume)
        volume = torch.from_numpy(volume)
        volume = volume.unsqueeze(3)
        volume = volume.permute(3, 0, 1, 2)
        label = torch.tensor(label, dtype=torch.float32)
        return volume, label

    def __len__(self):
        return len(self.fnames)


train_dataset = MosmedDataset(download=False,  train=True)
valid_dataset = MosmedDataset(download=False,  train=False)

train_loader = DataLoader(train_dataset, batch_size=4,
                          shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=4,
                          shuffle=True, num_workers=0)


class ConvNet3D(nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.maxpool = nn.MaxPool3d(kernel_size=2)

        self.conv1 = nn.Conv3d(
            in_channels=filters[0], out_channels=filters[1], kernel_size=3)
        self.bn1 = nn.BatchNorm3d(filters[1])

        self.conv2 = nn.Conv3d(
            in_channels=filters[1], out_channels=filters[2], kernel_size=3)
        self.bn2 = nn.BatchNorm3d(filters[2])

        self.conv3 = nn.Conv3d(
            in_channels=filters[2], out_channels=filters[3], kernel_size=3)
        self.bn3 = nn.BatchNorm3d(filters[3])

        self.conv4 = nn.Conv3d(
            in_channels=filters[3], out_channels=filters[4], kernel_size=3)
        self.bn4 = nn.BatchNorm3d(filters[4])

        self.gap = nn.AvgPool3d(kernel_size=2)
        self.dense = nn.Linear(256*3*3, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn1(self.maxpool(out))

        out = F.relu(self.conv2(out))
        out = self.bn2(self.maxpool(out))

        out = F.relu(self.conv3(out))
        out = self.bn3(self.maxpool(out))

        out = F.relu(self.conv4(out))
        out = self.bn4(self.maxpool(out))

        out = self.gap(out)
        out = self.dense(out.view(-1, 256*3*3))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))

        return out.view(-1)


filters = [1, 64, 64, 128, 256]
model = ConvNet3D(filters)

#sample_tensor = torch.zeros(4, 1, 128, 128, 64)
#out = model(sample_tensor)

lr = 0.0001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 15

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = model.to(device)
log_step = 1

decayRate = 0.96
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer, gamma=decayRate)


for epoch in range(epochs):
    model.train()
    train_running_loss = 0.0
    valid_running_loss = 0.0
    batch_count = 0
    lr_scheduler.step()

    print(f"Epoch: {epoch+1}")

    for i, batch in enumerate(train_loader):
        img, label = batch[0], batch[-1]
        img, label = img.to(device), label.to(device)
        output = model(img)

        optimizer.zero_grad()

        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()

        if i % log_step == 0:
            print(f"Batch: {i}, train loss: {train_running_loss / log_step}")
            train_running_loss = 0.0

    print(f"Train loss: {train_running_loss / len(train_dataset)}")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            img, label = batch[0], batch[-1]
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            valid_running_loss += loss.item()

            if i % log_step == 0:
                print(
                    f"Batch: {i}, train loss: {valid_running_loss / log_step}")
                valid_running_loss = 0.0

print("training finished")
