import os
import subprocess
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from zipfile import ZipFile
import pandas as pd
from PIL import Image
import torch


kaggle_biodatasets = [
    "aptos2019-blindness-detection",
    "chest-xray-pneumonia",
    "data-science-bowl-2018",
    "histopathologic-cancer-detection",
    "intel-mobileodt-cervical-cancer-screening",
    "skin-cancer-mnist"
]

def download_datasets(tag, path = "."):
    """[summary]

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
        bash_c_tag = ["kaggle", "datasets", "download", "-d", "paultimothymooney/chest-xray-pneumonia"]
    elif tag == "skin-cancer-mnist":
        bash_c_tag = ["kaggle", "datasets", "download", "-d", "kmader/skin-cancer-mnist-ham10000"]
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

def extract_zip(fzip, fnew):
    with ZipFile(fzip, 'r') as zip: 
        print('Extracting all the train files now...') 
        zip.extractall(fnew)
        print('Done!') 


class RetinopathyDataset(Dataset):
    """RetinopathyDataset from https://www.kaggle.com/c/aptos2019-blindness-detection/overview
    """
    def __init__(self, root: str, train: bool = True, download: bool = True):
        tag = "aptos2019-blindness-detection"

        if download:
            download_datasets(tag, path=root)

        extract_zip(os.path.join(root, tag+".zip"),os.path.join(root, tag))

        if train:
            self.csv_path = os.path.join(root, tag, "train.csv")
            self.img_path = os.path.join(root, tag, "train_images")

        self.data = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((256, 256), resample=Image.BILINEAR)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {   
            'image': transforms.ToTensor()(image),
            'labels': label
        }


class ChestXray(ImageFolder):
    """Chest X-ray dataset from

    Examples
    ----------
    >>> TEST = 'test'
    >>> VAL = 'val'
    >>> TRAIN = 'train'
    >>>
    >>> chestxray_dataset = {x: ChestXray(split=x) for x in [TRAIN, VAL, TEST]}
    >>>
    >>> dataloaders = {
    >>>     TRAIN: torch.utils.data.DataLoader(chestxray_dataset[TRAIN], batch_size = 4, shuffle=True),
    >>>     VAL: torch.utils.data.DataLoader(chestxray_dataset[VAL], batch_size = 1, shuffle=True),
    >>>     TEST: torch.utils.data.DataLoader(chestxray_dataset[TEST], batch_size = 1, shuffle=True)
    >>> }
    >>>
    >>> inputs, classes = next(iter(dataloaders[TRAIN]))
    """
    def __init__(self, root='/home/data/02_SSD4TB/suzy/datasets/public/chest-xray/', split='train', transform=None):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        super(ChestXray, self).__init__(
            root=os.path.join(root, split), transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
