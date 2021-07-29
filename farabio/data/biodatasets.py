import os
import random
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
<<<<<<< HEAD
from typing import Optional, Callable
=======
>>>>>>> main
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from zipfile import ZipFile


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

<<<<<<< HEAD
    Dataset is loaded using Kaggle API.
=======
    Dataset is loaded using Kaggle API. 
>>>>>>> main
    For further information on raw dataset and pneumonia detection, please refer to [1]_.

    Examples
    ----------
    >>> valid_dataset = ChestXrayDataset(root=_path, download=True, mode="val", show=True)

    .. image:: ../imgs/ChestXrayDataset.png
        :width: 300

    References
    ---------------
    .. [1] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    """

<<<<<<< HEAD
    def __init__(self, root: str = ".", download: bool = False, mode: str = "train", shape: int = 256, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, show: bool = True):
=======
    def __init__(self, root: str = ".", download: bool = False, mode: str = "train", shape: int = 256, transform: transforms = None, target_transform: transforms = None, show: int = 5):
>>>>>>> main
        tag = "chest-xray-pneumonia"

        modes = ["train", "val", "test"]
        assert mode in modes, "Available options for mode: train, val, test"

<<<<<<< HEAD
        self.shape = shape
=======
        self.target_transform = target_transform
>>>>>>> main

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))

        if transform is None:
            self.transform = self.default_transform(mode)
<<<<<<< HEAD
        else:
            self.transform = transform

        if target_transform is not None:
            self.target_transform = target_transform            

=======

>>>>>>> main
        if download:
            dataset_path = os.path.join(root, tag, "chest_xray", mode)
        else:
            dataset_path = os.path.join(root, mode)
<<<<<<< HEAD

        super(ChestXrayDataset, self).__init__(
            root=dataset_path, transform=self.transform)

        if show:
            self.visualize_batch()
=======

        super(ChestXrayDataset, self).__init__(
                root=dataset_path, transform=self.transform)

        self.visualize_dataset(show)
>>>>>>> main

    def __getitem__(self, index):
        path, target = self.samples[index]
        fname = path.split("/")[-1]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
<<<<<<< HEAD

        return sample, target, fname

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
            axs[0, i].set_title("..."+fnames[i][-11:-5])
=======

        return sample, target

    def default_transform(self, mode="train"):
        if mode == "train": 
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-20, +20)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

        elif mode == 'val' or mode == 'test':
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])

        return transform

    @staticmethod
    def imshow(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        fig = plt.gcf()
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
        return fig

    def visualize_dataset(self, show):
        loader = DataLoader(self, batch_size=show, shuffle=True)
        inputs, classes = next(iter(loader))
        class_names = self.classes
        out = torchvision.utils.make_grid(inputs)
        self.imshow(out, title=[class_names[x] for x in classes])
>>>>>>> main


class DSB18Dataset(Dataset):
    r"""PyTorch friendly DSB18Dataset class

    Dataset is loaded using Kaggle API.
    For further information on raw dataset and nuclei segmentation, please refer to [1]_.

    Examples
    ----------
    >>> train_dataset = DSB18Dataset(_path, transform=None, download=False, show=True)

    .. image:: ../imgs/DSB18Dataset.png
        :width: 300

    References
    ---------------
    .. [1] https://www.kaggle.com/c/data-science-bowl-2018/overview
    """

<<<<<<< HEAD
    def __init__(self, root: str = ".", download: bool = False, mode: str = "train", shape: int = 512, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, show: bool = True):

=======
    def __init__(self, root: str = ".", download: bool = False, mode: str = "train", shape: int = 512, transform: transform = None, target_transform: transforms = None, show: bool = True):
    
>>>>>>> main
        tag = "data-science-bowl-2018"
        modes = ["train", "val"]
        assert mode in modes, "Available options for mode: train, val"

        path = os.path.join(root, tag, "stage1_train")
        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"), os.path.join(root, tag))
            extract_zip(os.path.join(root, tag, "stage1_train.zip"), path)
        else:
            path = os.path.join(root, "stage1_train")

        self.path = path
        self.folders = os.listdir(self.path)
        self.shape = shape
<<<<<<< HEAD

=======
        
>>>>>>> main
        if transform is None:
            self.transform = self.default_transform()
        else:
            self.transform = transform
<<<<<<< HEAD

        if target_transform is None:
            self.target_transform = self.default_target_transform()
        else:
            self.target_transform = target_transform

=======
        
        if target_transform is None:
            self.target_transform = self.default_target_transform()
        else:
            self.target_transform = target_transform()
        
>>>>>>> main
        if show:
            self.visualize_batch()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
<<<<<<< HEAD

=======
        
>>>>>>> main
        fname = os.listdir(image_folder)[0]
        image_path = os.path.join(image_folder, fname)

        img = Image.open(image_path).convert('RGB')
        mask = self.get_mask(mask_folder)
<<<<<<< HEAD

        img = self.transform(img)
        mask = self.target_transform(mask)

        return img, mask, fname

=======
        
        img = self.transform(img)
        mask = self.target_transform(mask)
        
        return img, mask, fname
    
>>>>>>> main
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

<<<<<<< HEAD
    def default_transform(self):
        transform = transforms.Compose([
=======
        return transform
    
    def default_target_transform(self):
        target_transform = transforms.Compose([
>>>>>>> main
            transforms.ToTensor(),
            transforms.Resize((self.shape, self.shape))
        ])

<<<<<<< HEAD
        return transform

    def default_target_transform(self):
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.shape, self.shape))
        ])

        return target_transform

=======
        return target_transform
    
>>>>>>> main
    def visualize_batch(self):
        loader = DataLoader(self, shuffle=True, batch_size=4)
        imgs, masks, fnames = next(iter(loader))

        batch_inputs = convert_image_dtype(imgs, dtype=torch.uint8)
        batch_outputs = convert_image_dtype(masks, dtype=torch.bool)

        cells_with_masks = [
<<<<<<< HEAD
            draw_segmentation_masks(
                img, masks=mask, alpha=0.6, colors=(102, 255, 178))
            for img, mask in zip(batch_inputs, batch_outputs)
        ]

        self.show(cells_with_masks, fnames)
=======
            draw_segmentation_masks(img, masks=mask, alpha=0.6, colors = (102,255,178))
            for img, mask in zip(batch_inputs, batch_outputs)
        ]
>>>>>>> main

        self.show(cells_with_masks, fnames)
    
    @staticmethod
    def show(imgs, fnames):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
<<<<<<< HEAD
            axs[0, i].set_title("..."+fnames[i][-10:-4])
=======
            axs[0,i].set_title("..."+fnames[i][-10:-4])
>>>>>>> main


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

    def __init__(self, root: str = ".", mode: str = "train", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = True, show: bool = True):
        tag = "histopathologic-cancer-detection"

        modes = ["train", "val", "test"]
        assert mode in modes, "Available options for mode: train, val, test"

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))
            self.path = os.path.join(root, tag)
        else:
            self.path = os.path.join(root)

        if mode == "train":
            self.csv_path = os.path.join(self.path, "train_labels.csv")
            self.img_path = os.path.join(self.path, "train")
            self.labels = pd.read_csv(self.csv_path)
            train_data, val_data = train_test_split(
                self.labels, stratify=self.labels.label, test_size=0.1)
        else:
            self.img_path = os.path.join(self.path, "test")

        self.df = train_data.values

        if transform is None:
            self.transform = self.default_transform(mode)
        else:
            self.transform = transform

        if show:
            self.visualize_batch()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.img_path, img_name+'.tif')

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_name

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
        imgs, labels, fnames = next(iter(loader))

        list_imgs = [imgs[i] for i in range(len(imgs))]
        self.show(list_imgs, fnames, labels)

    @staticmethod
    def show(imgs, fnames, labels):
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
            if labels[i] == 0:
                lab = "non-tumor"
            else:
                lab = "tumor"
            axs[0, i].text(0, -0.2, str(int(labels[i])) + ": " +
                           lab, fontsize=12, transform=axs[0, i].transAxes)
            axs[0, i].set_title("..."+fnames[i][-6:])


class RANZCRDataset(Dataset):
    r"""RANZCR 2021 dataset class

    Catheters presence and position detection from RANZCR CLiP - Catheter and Line Position Challenge from [1]_

    Examples
    ----------
    >>> train_dataset = RANZCRDataset(".", train=True, transform=None, download=True)
    >>> train_dataset.visualize_dataset()

    .. image:: ../imgs/RANZCRDataset.png
        :width: 600

    References
    ---------------
    .. [1] https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data
    """

    def __init__(self, root: str, train: bool = True, transform=None, download: bool = False):
        tag = "ranzcr-clip-catheter-line-classification"

        if download:
            #download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))

        train_path = os.path.join(root, tag, "train")
        test_path = os  .path.join(root, tag, "test")

        # juggling
        data = pd.read_csv(os.path.join(root, tag, "train_annotations.csv"))
        data = data.drop(["data"], axis=1)

        # Converting the columns into integers.
        data_org = data['label']
        ord_enc = OrdinalEncoder()
        data[['label']] = ord_enc.fit_transform(data[['label']])

        # Converting the Labels from floats to integers.
        data.label = data.label.astype("int")

        # Grabbing the labels as a list.
        label = data["label"]
        label = label.to_list()

        seed = 42
        train_list = []

        for i in data.index:

            # Grabbing the file name.
            a = data["StudyInstanceUID"].loc[i]

            # Attaching the file's path to it.
            b = train_path + "/" + a + ".jpg"

            # Puttting it in a tupple along with it's label.
            train_list.append((b, data['label'].loc[i]))

        train_list, valid_list = train_test_split(train_list,
                                                  test_size=0.2,
                                                  random_state=seed)

        if transform is None:
            self.transforms = self.get_transform()

        if train:
            self.file_list = train_list
        else:
            self.file_list = valid_list

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):

        # Note that file list consists of tuples.
        # The first item in tuple is the image.
        img_path = self.file_list[idx][0]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transforms(img)

        # The second item in the tuple is the label.
        label = self.file_list[idx][1]

        return img_transformed, label

    def get_transform(self):
        """Default transform
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def visualize_dataset(self, n_images=5):
        random_idx = np.random.randint(1, len(self.file_list), size=16)
        fig, axes = plt.subplots(5, 5, figsize=(13, 13))

        for idx, ax in enumerate(axes.ravel()):
            img = Image.open(self.file_list[idx][0])
            ax.set_title(self.file_list[idx][-1])
            ax.imshow(img)

        # fig.savefig('RANZCRDataset.png')


class RetinopathyDataset(Dataset):
    r"""Retinopathy Dataset class

    Retina images taken using fundus photography from Kaggle APTOS 2019 Blindness Detection competition, [1]_.

    Examples
    ----------
    >>> train_dataset = RetinopathyDataset(root=".", transform=None, download=True)
    >>> train_dataset.visualize_dataset(9)

    .. image:: ../imgs/RetinopathyDataset.png
        :width: 300

    References
    ---------------
    .. [1] <https://www.kaggle.com/c/aptos2019-blindness-detection/data>`_
    """

    def __init__(self, root: str, train: bool = True, download: bool = True, transform=None):
        tag = "aptos2019-blindness-detection"

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))

        if train:
            self.csv_path = os.path.join(root, tag, "train.csv")
            self.img_path = os.path.join(root, tag, "train_images")

        self.data = pd.read_csv(self.csv_path)

        if transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.img_path, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((256, 256), resample=Image.BILINEAR)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])

        return {
            'image': self.transform(image),
            'labels': label
        }

    def visualize_dataset(self, n_images=9):
        """
        Function to visualize blindness images
        """
        train_csv = self.data
        fig = plt.figure(figsize=(30, 30))
        train_imgs = os.listdir(self.img_path)

        for idx, img in enumerate(np.random.choice(train_imgs, n_images)):
            ax = fig.add_subplot(3, n_images//3, idx+1, xticks=[], yticks=[])
            im = Image.open(os.path.join(self.img_path, img))
            plt.imshow(im)
            lab = train_csv.loc[train_csv['id_code'] ==
                                img.split('.')[0], 'diagnosis'].values[0]
            ax.set_title('Severity: %s' % lab, fontsize=40)
