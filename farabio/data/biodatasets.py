import os
import random
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensor
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from zipfile import ZipFile


kaggle_biodatasets = [
    "aptos2019-blindness-detection",
    "chest-xray-pneumonia",
    "data-science-bowl-2018",
    "histopathologic-cancer-detection",
    "intel-mobileodt-cervical-cancer-screening",
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
    """Chest X-ray dataset class

    Examples
    ----------
    >>> train_dataset = ChestXrayDataset(".", download=False)
    """

    def __init__(self, root: str, train: bool = True, shape: int = 256, transform=None, target_transform=None, download: bool = True):
        tag = "chest-xray-pneumonia"

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))

        self.target_transform = target_transform
        if transform is None:
            self.transform = self.get_train_transform(shape)

        if train:
            if download:
                train_path = os.path.join(
                    root, tag, "chest_xray", "train")
            else:
                train_path = os.path.join(
                    root, "chest-xray", "train")

        super(ChestXrayDataset, self).__init__(
            root=train_path, transform=self.transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def get_train_transform(self, img_shape):
        """Albumentations transform
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

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

    def visualize_dataset(self):
        """
        Function to visualize images and masks
        """
        train_dataset = ChestXrayDataset(
            "/home/data/02_SSD4TB/suzy/datasets/public/", download=False)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        inputs, classes = next(iter(train_loader))
        class_names = train_dataset.classes

        #inputs, classes = next(iter(train_loader))
        out = torchvision.utils.make_grid(inputs)
        self.imshow(out, title=[class_names[x] for x in classes])


class DSB18Dataset(Dataset):
    """Nuclei segmentation dataset class

    Examples
    ----------
    >>> train_dataset = DSB18Dataset(root=".", transform=None, download=True)
    """

    def __init__(self, root: str, train: bool = True, shape: int = 512, transform=None, download: bool = True):
        tag = "data-science-bowl-2018"

        path = os.path.join(root, tag, "stage1_train")
        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))
            extract_zip(os.path.join(root, tag, "stage1_train.zip"), path)

        if train:
            self.path = path

        if transform is None:
            self.transforms = self.get_train_transform(shape)

        self.folders = os.listdir(self.path)
        self.shape = shape

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
        fname = os.listdir(image_folder)[0]
        image_path = os.path.join(image_folder, fname)

        img = io.imread(image_path)[:, :, :3].astype('float32')
        img = transform.resize(img, (self.shape, self.shape))

        mask = self.get_mask(mask_folder, self.shape,
                             self.shape).astype('float32')

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        mask = mask[0].permute(2, 0, 1)
        return (img, mask, fname)

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)

        return mask

    def get_train_transform(self, img_shape):
        """Albumentations transform
        """
        return A.Compose(
            [
                A.Resize(img_shape, img_shape),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                A.HorizontalFlip(p=0.25),
                A.VerticalFlip(p=0.25),
                ToTensor()
            ])

    @staticmethod
    def format_image(img):
        img = np.array(np.transpose(img, (1, 2, 0)))
        mean = np.array((0.485, 0.456, 0.406))
        std = np.array((0.229, 0.224, 0.225))
        img = std * img + mean
        img = img*255
        img = img.astype(np.uint8)
        return img

    @staticmethod
    def format_mask(mask):
        mask = np.squeeze(np.transpose(mask, (1, 2, 0)))
        return mask

    def visualize_dataset(self, n_images, predict=None):
        """
        Function to visualize images and masks
        """
        images = random.sample(range(0, 670), n_images)
        figure, ax = plt.subplots(nrows=len(images), ncols=2, figsize=(5, 8))
        print(images)
        for i in range(0, len(images)):
            img_no = images[i]
            image, mask, fname = self.__getitem__(img_no)
            image = self.format_image(image)
            mask = self.format_mask(mask)
            ax[i, 0].imshow(image)
            ax[i, 1].imshow(mask, interpolation="nearest", cmap="gray")
            ax[i, 0].set_title("Ground Truth Image")
            ax[i, 1].set_title("Mask")
            ax[i, 0].set_axis_off()
            ax[i, 1].set_axis_off()
            plt.tight_layout()

        return plt

# train_dataset = DSB18Dataset(root="/home/data/02_SSD4TB/suzy/datasets/public/", transform=None, download=False)


class RetinopathyDataset(Dataset):
    """Retinopathy Dataset class

    Examples
    ----------
    >>> train_dataset = RetinopathyDataset(root=".", transform=None)
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


class HistocancerDataset(Dataset):
    """Histopathologic Cancer Dataset class

    Examples
    ----------
    >>> train_dataset = HistocancerDataset(root="./", download=None, train=True)
    """

    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True):
        tag = "histopathologic-cancer-detection"

        if download:
            download_datasets(tag, path=root)
            extract_zip(os.path.join(root, tag+".zip"),
                        os.path.join(root, tag))

        if train:
            self.csv_path = os.path.join(root, tag, "train_labels.csv")
            self.img_path = os.path.join(root, tag, "train")
            self.labels = pd.read_csv(self.csv_path)
            train_data, val_data = train_test_split(
                self.labels, stratify=self.labels.label, test_size=0.1)
        else:
            self.img_path = os.path.join(root, tag, "test")

        self.df = train_data.values

        if transform is None:
            self.transform = self.get_train_transform
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name+'.tif')
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @staticmethod
    def get_train_transform(self):
        """Default transform for training data
        """
        return transforms.Compose([transforms.ToPILImage(),
                                   transforms.Pad(64, padding_mode='reflect'),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.RandomRotation(20),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    @staticmethod
    def get_valid_transform(self):
        """Default transform for validation data
        """
        return transforms.Compose([transforms.ToPILImage(),
                                   transforms.Pad(64, padding_mode='reflect'),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def visualize_dataset(self, n_images=10):
        fig = plt.figure(figsize=(12, 4))

        train_imgs = os.listdir(self.img_path)
        for idx, img in enumerate(np.random.choice(train_imgs, n_images)):
            ax = fig.add_subplot(2, n_images//2, idx+1, xticks=[], yticks=[])
            im = Image.open(os.path.join(self.img_path, img))
            plt.imshow(im)
            lab = self.labels.loc[self.labels['id'] ==
                                  img.split('.')[0], 'label'].values[0]
            if lab == 1:
                ax.set_title(f'{lab} = tumor')
            else:
                ax.set_title(f'{lab} = non-tumor')
