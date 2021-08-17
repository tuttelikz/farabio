import os
import unittest
import subprocess
import numpy as np
import pandas as pd
from zipfile import ZipFile
from PIL import Image
from typing import Optional, Callable
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import draw_segmentation_masks

__all__ = ['ChestXrayDataset', 'DSB18Dataset', 'HistocancerDataset', 
'RANZCRDataset', 'RetinopathyDataset']


kaggle_biodatasets = [
    "aptos2019-blindness-detection",
    "chest-xray-pneumonia",
    "data-science-bowl-2018",
    "histopathologic-cancer-detection",
    "intel-mobileodt-cervical-cancer-screening",
    "ranzcr-clip-catheter-line-classification",
    "skin-cancer-mnist"
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
                "skin-cancer-mnist"
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


class TestBiodatasets(unittest.TestCase):
    def testChestXrayDataset(self):
        _path = "/home/data/02_SSD4TB/suzy/datasets/public/chest-xray"
        valid_dataset = ChestXrayDataset(
            root=_path, download=False, mode="val", show=False)
        print(valid_dataset)

    def testDSB18Dataset(self):
        _path = "/home/data/02_SSD4TB/suzy/datasets/public/data-science-bowl-2018"
        train_dataset = DSB18Dataset(
            root=_path, transform=None, mode="train", download=False, show=False)
        print(train_dataset)

    def testHistocancerDataset(self):
        _path = "/home/data/02_SSD4TB/suzy/datasets/public/histopathologic-cancer-detection"
        train_dataset = HistocancerDataset(
            root=_path, download=False, mode="train", show=False)
        print(train_dataset)

    def testRANZCRDataset(self):
        _path = "/home/data/02_SSD4TB/suzy/datasets/public/ranzcr-clip-catheter-line-classification"
        train_dataset = RANZCRDataset(
            root=_path, show=False, shape=512, mode="train", download=False)
        print(train_dataset)

    def testRetinopathyDataset(self):
        _path = "/home/data/02_SSD4TB/suzy/datasets/public/aptos2019-blindness-detection"
        train_dataset = RetinopathyDataset(
            root=_path, mode="train", show=False, download=False)
        print(train_dataset)


# unittest.main()
