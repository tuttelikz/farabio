import os
import unittest
import subprocess
import numpy as np
import pandas as pd
from zipfile import ZipFile
from PIL import Image
from typing import Optional, Callable
import matplotlib.pyplot as plt
import pydicom
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import requests
import io
import wget
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F


__all__ = ['ChestXrayDataset', 'DSB18Dataset', 'HistocancerDataset',
           'RANZCRDataset', 'RetinopathyDataset', 'VinBigDataset']


kaggle_biodatasets = [
    "aptos2019-blindness-detection",
    "chest-xray-pneumonia",
    "data-science-bowl-2018",
    "histopathologic-cancer-detection",
    "intel-mobileodt-cervical-cancer-screening",
    "ranzcr-clip-catheter-line-classification",
    "skin-cancer-mnist",
    "vinbigdata-chest-xray-abnormalities-detection"
]


def download_datasets(tag, path="."):
    """Helper function to download datasets

    Parameters
    ----------
    tag : str
        tag for dataset

        .. note::
            available tags:
            kaggle_biodatasets = [
                "aptos2019-blindness-detection",
                "chest-xray-pneumonia",
                "data-science-bowl-2018",
                "histopathologic-cancer-detection",
                "intel-mobileodt-cervical-cancer-screening",
                "ranzcr-clip-catheter-line-classification",
                "skin-cancer-mnist",
                "vinbigdata-chest-xray-abnormalities-detection"
            ]
    path : str, optional
        path where to save dataset, by default "."
    Examples
    ----------
    >>> download_datasets(tag="skin-cancer-mnist", path=".")
    """
    if tag == "chest-xray-pneumonia":
        bash_c_tag = ["kaggle", "datasets", "download",
                      "-d", "paultimothymooney/chest-xray-pneumonia"]
    elif tag == "skin-cancer-mnist":
        bash_c_tag = ["kaggle", "datasets", "download",
                      "-d", "kmader/skin-cancer-mnist-ham10000"]
    else:
        bash_c = ["kaggle", "competitions", "download", "-c"]
        bash_c_tag = bash_c.copy()
        bash_c_tag.append(tag)

    prev_cwd = os.getcwd()
    os.chdir(path)
    process = subprocess.Popen(bash_c_tag, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)

    os.chdir(prev_cwd)


def extract_zip(fzip, fnew=None):
    with ZipFile(fzip, 'r') as zip:  # ZipFile(fzip, 'r') as zip:
        print('Extracting all the train files now...')
        zip.extractall(fnew)
        print('Done!')


class ChestXrayDataset(ImageFolder):
    r"""PyTorch friendly ChestXrayDataset class

    Dataset is loaded using Kaggle API.
    For further information on raw dataset and pneumonia detection, please refer to [1]_.

    Examples
    ----------
    >>> valid_dataset = ChestXrayDataset(root=_path, download=True, mode="val", show=True)

    .. image:: ../imgs/ChestXrayDataset.png
        :width: 600

    References
    ---------------
    .. [1] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    """

    def __init__(self, root: str = ".", download: bool = False, mode: str = "train", shape: int = 256, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, show: bool = True):
        tag = "chest-xray-pneumonia"

        modes = ["train", "val", "test"]
        assert mode in modes, "Available options for mode: train, val, test"

        self.shape = shape
        self.mode = mode
        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))

        if transform is None:
            self.transform = self.default_transform(mode)
        else:
            self.transform = transform

        if target_transform is not None:
            self.target_transform = target_transform

        if download:
            dataset_path = os.path.join(root, tag, "chest_xray", mode)
        else:
            dataset_path = os.path.join(root, mode)

        super(ChestXrayDataset, self).__init__(
            root=dataset_path, transform=self.transform)

        if show:
            self.visualize_batch()

    def __getitem__(self, index):
        path, target = self.samples[index]
        fname = path.split("/")[-1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, fname

    def default_transform(self, mode="train"):
        if mode == "train":
            transform = transforms.Compose([
                transforms.Resize((self.shape, self.shape)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

        elif mode == 'val' or mode == 'test':
            transform = transforms.Compose([
                transforms.Resize((self.shape, self.shape)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

        return transform

    def visualize_batch(self):
        loader = DataLoader(self, batch_size=4, shuffle=True)
        inputs, labels, fnames = next(iter(loader))
        list_imgs = [inputs[i] for i in range(len(inputs))]

        self.show(list_imgs, labels, fnames)

    def show(self, imgs, labels, fnames):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):

            img = img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * img + mean
            inp = np.clip(inp, 0, 1)

            axs[0, i].imshow(np.asarray(inp))
            axs[0, i].set(xticks=[], yticks=[])
            axs[0, i].text(0, -0.2, str(int(labels[i])) + ": " +
                           self.classes[labels[i]], transform=axs[0, i].transAxes)
            axs[0, i].set_title("..."+fnames[i][-12:-5])


class DSB18Dataset(Dataset):
    r"""PyTorch friendly DSB18Dataset class

    Dataset is loaded using Kaggle API.
    For further information on raw dataset and nuclei segmentation, please refer to [1]_.

    Examples
    ----------
    >>> train_dataset = DSB18Dataset(_path, transform=None, download=False, show=True)

    .. image:: ../imgs/DSB18Dataset.png
        :width: 600

    References
    ---------------
    .. [1] https://www.kaggle.com/c/data-science-bowl-2018/overview
    """

    def __init__(self, root: str = ".", download: bool = False, mode: str = "train", shape: int = 512, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, show: bool = True):

        tag = "data-science-bowl-2018"
        modes = ["train", "val", "test"]
        assert mode in modes, "Available options for mode: train, val"

        if mode == "train" or mode == "val":
            stage = "stage1_train"
        else:
            stage = "stage1_test"

        self.mode = mode
        path = os.path.join(root, tag, stage)

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))
            extract_zip(os.path.join(root, tag, stage + ".zip"), path)
        else:
            path = os.path.join(root, stage)

        self.path = path
        self.shape = shape

        if self.mode != "test":
            seed = 42
            train_list = os.listdir(self.path)
            train_list, valid_list = train_test_split(
                train_list,
                test_size=0.2,
                random_state=seed
            )

            if self.mode == "train":
                self.folders = train_list
            elif self.mode == "val":
                self.folders = valid_list
        else:
            self.folders = os.listdir(self.path)

        if transform is None:
            self.transform = self.default_transform()
        else:
            self.transform = transform

        if target_transform is None:
            self.target_transform = self.default_target_transform()
        else:
            self.target_transform = target_transform

        if show:
            self.visualize_batch()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):

        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        fname = os.listdir(image_folder)[0]
        image_path = os.path.join(image_folder, fname)
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        if self.mode != "test":
            mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
            mask = self.get_mask(mask_folder)
            mask = self.target_transform(mask)
            sample = (img, mask, fname)
        else:
            sample = (img, fname)

        return sample

    def get_mask(self, mask_folder):
        mask = np.zeros((self.shape, self.shape, 1), dtype=bool)
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            mask_ = transform.resize(mask_, (self.shape, self.shape))
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)

        return mask

    def default_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.shape, self.shape))
        ])

        return transform

    def default_target_transform(self):
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.shape, self.shape))
        ])

        return target_transform

    def visualize_batch(self):
        loader = DataLoader(self, shuffle=True, batch_size=4)

        if self.mode != "test":
            imgs, masks, fnames = next(iter(loader))
        else:
            imgs, fnames = next(iter(loader))

        batch_inputs = F.convert_image_dtype(imgs, dtype=torch.uint8)

        if self.mode != "test":
            batch_outputs = F.convert_image_dtype(masks, dtype=torch.bool)
            list_imgs = [
                draw_segmentation_masks(
                    img, masks=mask, alpha=0.6, colors=(102, 255, 178))
                for img, mask in zip(batch_inputs, batch_outputs)
            ]
        else:
            list_imgs = [imgs[i] for i in range(len(imgs))]

        self.show(list_imgs, fnames)

    def show(self, imgs, fnames):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[0, i].set_title("..."+fnames[i][-10:-4])


class HistocancerDataset(Dataset):
    r"""PyTorch friendly HistocancerDataset class

     Dataset is loaded using Kaggle API.
     For further information on raw dataset and tumor classification, please refer to [1]_.

     Examples
     ----------
     >>> train_dataset = HistocancerDataset(root=".", download=False, mode="train")

     .. image:: ../imgs/HistocancerDataset.png
         :width: 600

     References
     ---------------
     .. [1] <https://www.kaggle.com/c/histopathologic-cancer-detection/data>`_
     """

    def __init__(self, root: str = ".", mode: str = "train", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, show: bool = True):
        tag = "histopathologic-cancer-detection"

        modes = ["train", "val", "test"]
        assert mode in modes, "Available options for mode: train, val, test"

        self.mode = mode

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))
            self.path = os.path.join(root, tag)
        else:
            self.path = os.path.join(root)

        if self.mode != "test":
            self.csv_path = os.path.join(self.path, "train_labels.csv")
            self.img_path = os.path.join(self.path, "train")
            self.labels = pd.read_csv(self.csv_path)
            train_data, val_data = train_test_split(
                self.labels, stratify=self.labels.label, test_size=0.1)

            if self.mode == "train":
                data = train_data
            elif self.mode == "val":
                data = val_data

            self.data = data.values
        else:
            self.img_path = os.path.join(self.path, "test")
            self.data = os.listdir(self.img_path)

        if transform is None:
            self.transform = self.default_transform(mode)
        else:
            self.transform = transform

        self.target_transform = target_transform

        if show:
            self.visualize_batch()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if self.mode != "test":
            fname, label = self.data[index]
            img_path = os.path.join(self.img_path, fname+'.tif')
        else:
            fname = self.data[index]
            img_path = os.path.join(self.img_path, fname)

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.transform(label)

        if self.mode != "test":
            sample = (img, label, fname)
        else:
            sample = (img, fname)

        return sample

    def default_transform(self, mode):
        if mode == "train":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

        elif mode == "val" or mode == "test":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

        return transform

    def visualize_batch(self):
        loader = DataLoader(self, batch_size=4, shuffle=True)

        if self.mode != "test":
            imgs, labels, fnames = next(iter(loader))
        else:
            imgs, fnames = next(iter(loader))
            labels = None

        list_imgs = [imgs[i] for i in range(len(imgs))]
        self.show(list_imgs, fnames, labels)

    def show(self, imgs, fnames, labels=None):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * img + mean
            inp = np.clip(inp, 0, 1)

            axs[0, i].imshow(np.asarray(inp))
            axs[0, i].set(xticklabels=[], yticklabels=[],
                          xticks=[], yticks=[])

            if self.mode != "test":
                if labels[i] == 0:
                    lab = "non-tumor"
                else:
                    lab = "tumor"
                axs[0, i].set_title("..."+fnames[i][-6:])
                axs[0, i].text(0, -0.2, str(int(labels[i])) + ": " +
                               lab, transform=axs[0, i].transAxes)
            else:
                axs[0, i].set_title("..."+fnames[i][-11:-4])


class RANZCRDataset(Dataset):
    r"""PyTorch friendly RANZCRDataset class

    Dataset is loaded using Kaggle API.
    For further information on raw dataset and catheters presence, please refer to [1]_.

    Examples
    ----------
    >>> train_dataset = RANZCRDataset(_path_ranzcr, show=True, shape=512)

    .. image:: ../imgs/RANZCRDataset.png
        :width: 600

    References
    ---------------
    .. [1] https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data
    """

    def __init__(self, root: str = ".", mode: str = "train", shape: int = 256, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, show: bool = True):
        tag = "ranzcr-clip-catheter-line-classification"

        modes = ["train", "val", "test"]
        assert mode in modes, "Available options for mode: train, val, test"

        self.mode = mode

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))
            path = os.path.join(root, tag)
        else:
            path = root

        train_path = os.path.join(path, "train")
        test_path = os.path.join(path, "test")
        csv_path = os.path.join(path, "train_annotations.csv")

        self.data = pd.read_csv(csv_path)
        self.labels, self.encoded_labels = self.get_labels()
        self.train_list, self.valid_list = self.get_train_valid(train_path)
        self.shape = shape

        if transform is None:
            self.transform = self.default_transform()
        else:
            self.transform = transform

        if target_transform is not None:
            self.target_transform = target_transform

        if self.mode != "test":
            if self.mode == "train":
                self.file_list = self.train_list
            else:
                self.file_list = self.valid_list
        else:
            self.file_list = glob.glob(test_path+"/*")

        if show:
            self.visualize_batch()

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        if self.mode != "test":
            img_path = self.file_list[idx][0]
            fname = img_path.split("/")[-1]
        else:
            img_path = self.file_list[idx]
            fname = img_path.split("/")[-1]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if self.mode != "test":
            label = self.file_list[idx][1]
            sample = (img, label, fname)
        else:
            sample = (img, fname)

        return sample

    def default_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.shape, self.shape)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        return transform

    def visualize_batch(self):
        loader = DataLoader(self, batch_size=4, shuffle=True)

        if self.mode != "test":
            imgs, labels, fnames = next(iter(loader))
        else:
            imgs, fnames = next(iter(loader))
            labels = None

        list_imgs = [imgs[i] for i in range(len(imgs))]
        self.show(list_imgs, fnames, labels)

    def show(self, imgs, fnames, labels=None):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * img + mean
            inp = np.clip(inp, 0, 1)

            axs[0, i].imshow(np.asarray(inp))
            axs[0, i].set(xticklabels=[], yticklabels=[],
                          xticks=[], yticks=[])
            axs[0, i].set_title("..."+fnames[i][-11:-4])

            if self.mode != "test":
                lab = self.unique_labels[labels[i]]
                axs[0, i].text(0, -0.2, str(int(labels[i])) +
                               ": " + lab, transform=axs[0, i].transAxes)

    def get_labels(self):
        self.data = self.data.drop(["data"], axis=1)

        data_org = self.data['label']
        labels = data_org.to_list()

        used = set()
        self.unique_labels = [
            x for x in labels if x not in used and (used.add(x) or True)]

        ord_enc = OrdinalEncoder()
        self.data[['label']] = ord_enc.fit_transform(self.data[['label']])

        self.data.label = self.data.label.astype("int")

        label = self.data["label"]
        label = label.to_list()

        encoded_labels = label
        return labels, encoded_labels

    def get_train_valid(self, train_path):
        seed = 42
        train_list = []

        for i in self.data.index:
            a = self.data["StudyInstanceUID"].loc[i]
            b = train_path + "/" + a + ".jpg"
            train_list.append((b, self.data['label'].loc[i]))

        train_list, valid_list = train_test_split(train_list,
                                                  test_size=0.2,
                                                  random_state=seed)
        return train_list, valid_list


class RetinopathyDataset(Dataset):
    r"""PyTorch friendly RetinopathyDataset class

    Dataset is loaded using Kaggle API.
    For further information on raw dataset and blindness detection, please refer to [1]_.

    Examples
    ----------
    >>> train_dataset = RetinopathyDataset(".", mode="train", show=True)

    .. image:: ../imgs/RetinopathyDataset.png
        :width: 600

    References
    ---------------
    .. [1] <https://www.kaggle.com/c/aptos2019-blindness-detection/data>`_
    """

    def __init__(self, root: str = ".", mode: str = "train", shape: int = 256, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, show: bool = True):
        tag = "aptos2019-blindness-detection"

        if download:
            download_datasets(tag, path=root)

            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))
            path = os.path.join(root, tag)
        else:
            path = root

        self.mode = mode

        if mode != "test":
            self.csv_path = os.path.join(path, "train.csv")
            self.img_path = os.path.join(path, "train_images")
            data = pd.read_csv(self.csv_path)

            train_idx, val_idx = train_test_split(
                range(len(data)), test_size=0.1,)

            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]

            if self.mode == "train":
                self.data = train_data
                self.data.reset_index(drop=True, inplace=True)
            else:
                self.data = val_data
                self.data.reset_index(drop=True, inplace=True)
        else:
            self.img_path = os.path.join(path, "test_images")
            self.data = os.listdir(self.img_path)

        self.shape = shape

        if transform is None:
            self.transform = self.default_transform()
        else:
            self.transform = transform

        if target_transform is not None:
            self.target_transform = target_transform

        if show:
            self.visualize_batch()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode != "test":
            fname = self.data.loc[idx, 'id_code'] + ".png"
        else:
            fname = self.data[idx]

        img_name = os.path.join(self.img_path, fname)

        img = Image.open(img_name).convert("RGB")
        img = self.transform(img)

        if self.mode != "test":
            label = torch.tensor(self.data.loc[idx, 'diagnosis'])
            sample = (img, label, fname)
        else:
            sample = (img, fname)

        return sample

    def default_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.shape, self.shape)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform

    def visualize_batch(self):
        loader = DataLoader(self, batch_size=4, shuffle=True)

        if self.mode != "test":
            imgs, labels, fnames = next(iter(loader))
        else:
            imgs, fnames = next(iter(loader))
            labels = None

        list_imgs = [imgs[i] for i in range(len(imgs))]
        self.show(list_imgs, fnames, labels)

    def show(self, imgs, fnames, labels=None):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * img + mean
            inp = np.clip(inp, 0, 1)

            axs[0, i].imshow(np.asarray(inp))
            axs[0, i].set(xticklabels=[], yticklabels=[],
                          xticks=[], yticks=[])
            axs[0, i].set_title("..."+fnames[i][-10:-4])

            if self.mode != "test":
                axs[0, i].text(0, -0.2, "Severity: " +
                               str(int(labels[i])), transform=axs[0, i].transAxes)


class VinBigDataset(Dataset):
    r"""PyTorch friendly VinBigDataset class

    Dataset is loaded using Kaggle API.
    For further information on raw dataset and nuclei segmentation, please refer to [1]_.

    Examples
    ----------
    >>> train_dataset = VinBigDataset(_path, transform=None, download=False, mode="train", show=True)

    .. image:: ../imgs/DSB18Dataset.png
        :width: 600

    References
    ---------------
    .. [1] https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
    """

    def __init__(self, root: str = ".", download: bool = False, mode: str = "train", shape: int = 512, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, show: bool = True):
        tag = "vinbigdata-chest-xray-abnormalities-detection"

        modes = ["train", "val", "test"]
        assert mode in modes, "Available options for mode: train, val, test"

        self.shape = shape
        self.mode = mode

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))
            DIR_INPUT = os.path.join(root, tag)
        else:
            DIR_INPUT = root

        DIR_TRAIN = f'{DIR_INPUT}/train'
        DIR_TEST = f'{DIR_INPUT}/test'

        train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
        (train_df, valid_df) = self._split(train_df)
        (train_df, valid_df) = self._preprocess(train_df, valid_df)

        self._init_labels()
        self.num_classes = len(self.id_class.values())

        if self.mode == "train":
            self.image_ids = train_df["image_id"].unique()
            self.df = train_df
            self.image_dir = DIR_TRAIN
        elif self.mode == "val":
            self.image_ids = valid_df["image_id"].unique()
            self.df = valid_df
            self.image_dir = DIR_TRAIN
        else:
            print("Test case not handled")

        if transform is None:
            self.transforms = self.default_transform(mode)
        else:
            self.transforms = transform

        if show:
            self.visualize_batch()

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[(self.df['image_id'] == image_id)]
        records = records.reset_index(drop=True)
        dicom = pydicom.dcmread(f"{self.image_dir}/{image_id}.dicom")
        image = dicom.pixel_array

        # this part was only in train
        if "PhotometricInterpretation" in dicom:
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                image = np.amax(image) - image

        intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else 0.0
        slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1.0

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        image = np.stack([image, image, image])
        image = image.astype('float32')
        image = image - image.min()
        image = image / image.max()
        image = image * 255.0
        image = image.transpose(1, 2, 0)

        if self.mode == "train" or self.mode == "val":
            if records.loc[0, "class_id"] == 0:
                records = records.loc[[0], :]

            boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
            labels = torch.tensor(
                records["class_id"].values, dtype=torch.int64)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            target['image_id'] = torch.tensor([index])
            target['area'] = area
            target['iscrowd'] = iscrowd

            if self.transforms:
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transforms(**sample)
                image = sample['image']

                target['boxes'] = torch.tensor(
                    sample['bboxes']).type(torch.float32)

            if target["boxes"].shape[0] == 0:
                # Albumentation cuts the target (class 14, 1x1px in the corner)
                target["boxes"] = torch.from_numpy(
                    np.array([[0.0, 0.0, 1.0, 1.0]])).type(torch.float32)
                target["area"] = torch.tensor([1.0], dtype=torch.float32)
                target["labels"] = torch.tensor([0], dtype=torch.int64)

            return image, target, image_id
        else:
            if self.transforms:
                sample = {
                    'image': image,
                }
                sample = self.transforms(**sample)
                image = sample['image']

            return image, image_id

    def __len__(self):
        return self.image_ids.shape[0]

    def default_transform(self, mode="train"):
        if mode == "train":
            transform = A.Compose([
                A.Flip(0.5),
                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.25),
                A.LongestMaxSize(max_size=800, p=1.0),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        elif mode == 'val':
            transform = A.Compose([
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        else:
            transform = A.Compose([
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1),
                            max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)
            ])

        return transform

    def _split(self, train_df):
        train_df.fillna(0, inplace=True)
        train_df.loc[train_df["class_id"] == 14, ['x_max', 'y_max']] = 1.0

        # FasterRCNN handles class_id==0 as the background.
        train_df["class_id"] = train_df["class_id"] + 1
        train_df.loc[train_df["class_id"] == 15, ["class_id"]] = 0

        image_ids = train_df['image_id'].unique()
        valid_ids = image_ids[-10000:]
        train_ids = image_ids[:-10000]

        valid_df = train_df[train_df['image_id'].isin(valid_ids)]
        train_df = train_df[train_df['image_id'].isin(train_ids)]

        train_df["class_id"] = train_df["class_id"].apply(lambda x: x+1)
        valid_df["class_id"] = valid_df["class_id"].apply(lambda x: x+1)

        return (train_df, valid_df)

    def _preprocess(self, train_df, valid_df):
        train_df['area'] = (train_df['x_max'] - train_df['x_min']) * \
            (train_df['y_max'] - train_df['y_min'])
        valid_df['area'] = (valid_df['x_max'] - valid_df['x_min']) * \
            (valid_df['y_max'] - valid_df['y_min'])
        train_df = train_df[train_df['area'] > 1]
        valid_df = valid_df[valid_df['area'] > 1]

        train_df = train_df[(train_df['class_id'] > 1) &
                            (train_df['class_id'] < 15)]
        valid_df = valid_df[(valid_df['class_id'] > 1) &
                            (valid_df['class_id'] < 15)]

        train_df = train_df.drop(['area'], axis=1)

        return (train_df, valid_df)

    def _init_labels(self):
        self.id_class = {
            0: "Aortic enlargement",
            1: "Atelectasis",
            2: "Calcification",
            3: "Cardiomegaly",
            4: "Consolidation",
            5: "ILD",
            6: "Infiltration",
            7: "Lung Opacity",
            8: "Nodule/Mass",
            9: "Other lesion",
            10: "Pleural effusion",
            11: "Pleural thickening",
            12: "Pneumothorax",
            13: "Pulmonary fibrosis"
        }

        self.id_clr = {}

        for j, _clr in enumerate(sns.color_palette(n_colors=len(self.id_class.keys()))):
            self.id_clr[j] = tuple(np.uint8(255*np.array(_clr)))

    def _label_to_name(self, id):
        id = int(id)
        id = id-1

        if id in self.id_class:
            return self.id_class[id]
        else:
            return str(id)

    def _collate_fn(self, batch):
        return tuple(zip(*batch))

    def visualize_batch(self):
        if self.mode == "train":
            loader = DataLoader(
                self,
                batch_size=4,
                shuffle=True,
                num_workers=4,
                collate_fn=self._collate_fn
            )
        else:
            loader = DataLoader(
                self,
                batch_size=4,
                shuffle=True,
                num_workers=4,
                collate_fn=self._collate_fn
            )

        images, targets, image_ids = next(iter(loader))

        bbx_ = []

        for i, img in enumerate(images):
            img_int = img * 255
            img_int = img_int.type(torch.uint8)

            bboxes_int = targets[i]['boxes'].type(torch.uint8)

            bbclrs = []
            bbclss = []

            for jj, label in enumerate(targets[i]['labels']):
                bbclrs.append(self.id_clr[int(label)])
                bbclss.append(self._label_to_name(int(label)))

            bbx_.append(draw_bounding_boxes(img_int, boxes=bboxes_int,
                                            colors=bbclrs, font_size=20, labels=bbclss))

        self._show(bbx_)

    def _show(self, imgs):
        plt.rcParams["savefig.bbox"] = 'tight'

        if not isinstance(imgs, list):
            imgs = [imgs]

        fix, axs = plt.subplots(
            ncols=len(imgs), squeeze=False, figsize=(50, 50))

        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class TestBiodatasets(unittest.TestCase):
    def testChestXrayDataset(self):
        _path = "/home/data/07_SSD4TB/public-datasets/chest-xray"
        valid_dataset = ChestXrayDataset(
            root=_path, download=False, mode="val", show=False)
        print(valid_dataset)

    def testDSB18Dataset(self):
        _path = "/home/data/07_SSD4TB/public-datasets/data-science-bowl-2018"
        train_dataset = DSB18Dataset(
            root=_path, transform=None, mode="train", download=False, show=False)
        print(train_dataset)

    def testHistocancerDataset(self):
        _path = "/home/data/07_SSD4TB/public-datasets/histopathologic-cancer-detection"
        train_dataset = HistocancerDataset(
            root=_path, download=False, mode="train", show=False)
        print(train_dataset)

    def testRANZCRDataset(self):
        _path = "/home/data/07_SSD4TB/public-datasets/ranzcr-clip-catheter-line-classification"
        train_dataset = RANZCRDataset(
            root=_path, show=False, shape=512, mode="train", download=False)
        print(train_dataset)

    def testRetinopathyDataset(self):
        _path = "/home/data/07_SSD4TB/public-datasets/aptos2019-blindness-detection"
        train_dataset = RetinopathyDataset(
            root=_path, mode="train", show=False, download=False)
        print(train_dataset)

    def testVinBigDataset(self):
        _path = "/home/data/07_SSD4TB/public-datasets/vinbigdata-chest-xray-abnormalities-detection"
        train_dataset = VinBigDataset(
            root=_path, mode="train", show=False, download=False)
        print(train_dataset)


class MosmedDataset(Dataset):
    def __init__(self, download: bool = False, save_path: str = ".", train: bool = True):
        if download:
            ct_link = {
                "CT-0": "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip",
                "CT-23": "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
            }
            for key in ct_link.keys():
                wget.download(ct_link[key], save_path)
                with ZipFile(os.path.join(save_path, ct_link + ".zip"), "r") as z_fp:
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

# unittest.main()
