from torchvision import transforms
from farabio.prep.transforms import Normalize, ToTensor, ImgToTensor, ImgNormalize, ToLongTensor
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch
import numpy as np
from farabio.data.datasets import XenopusDataset, ImgLabelDataset
from farabio.prep.transforms import Compose, Dataraf, Datajit
from farabio.utils.helpers import _matches, calc_weights


def get_trainloader(config, pth_train_img, pth_train_lbl, augment=False):
    random_seed = config.random_seed
    batch_size = config.batch_size
    validation_split = config.val_split
    shuffle_dataset = config.shuffle_data
    workers = config.num_workers
    balanced = config.balanced

    if augment is True:
        augs = {
            "train": Compose([
                Datajit(br=0.2, cnt=0.2),
                Dataraf(deg=90, trn=(0.2, 0.2), sc=(0.8, 1.2), shr=20),
                ImgToTensor()  # normalize?
            ]),
            "valid": ImgToTensor()
        }

        transformed_train_data = ImgLabelDataset(
            img_dir=pth_train_img,
            lbl_dir=pth_train_lbl,
            augs=augs["train"],
        )

    if augment is False:
        # composed_train = transforms.Compose([Normalize(), ToTensor()])
        # transformed_train_data = XenopusDataset(img_dir=pth_train_img,
        #                                         transform=composed_train,
        #                                         lbl_dir=pth_train_lbl,
        #                                         mode="train")
        # not sure if transform compose is same as transforms compose
        composed_train = transforms.Compose(
            [ImgNormalize(), ToLongTensor()])
        transformed_train_data = XenopusDataset(img_dir=pth_train_img,
                                                transform=composed_train,
                                                lbl_dir=pth_train_lbl,
                                                mode="train")

        dataset_size = len(transformed_train_data)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            transformed_train_data, batch_size=batch_size, sampler=train_sampler, num_workers=workers)

        valid_loader = torch.utils.data.DataLoader(
            transformed_train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=workers)