from torchvision import transforms
from farabi.data.transforms import Normalize, ToTensor, ImgToTensor, ImgNormalize, ToLongTensor
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch
import numpy as np
from farabi.data.datasets import XenopusDataset, ImgLabelDataset
from farabi.data.transforms import Compose, Dataraf, Datajit
from farabi.utils.helpers import _matches, calc_weights


def get_trainloader(config, pth_train_img, pth_train_lbl, augment=False):
    random_seed = config.random_seed
    batch_size = config.batch_size
    validation_split = config.val_split
    shuffle_dataset = config.shuffle_data
    workers = config.num_workers
    balanced = config.balanced

    if balanced is True:
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

    elif balanced is False:

        if augment is True:
            augs = {
                "train": Compose([
                    Datajit(br=0.2, cnt=0.2),
                    Dataraf(deg=90, trn=(0.2, 0.2), sc=(0.8, 1.2), shr=20),
                    ImgNormalize(),
                    ToLongTensor()
                ]),
                "valid": Compose([
                    ImgNormalize(),
                    ToLongTensor()
                ])
            }

            transformed_train_data = ImgLabelDataset(
                img_dir=pth_train_img,
                lbl_dir=pth_train_lbl,
                augs=augs["train"],
            )

        if augment is False:
            composed_train = transforms.Compose(
                [ImgNormalize(), ToLongTensor()])
            transformed_train_data = XenopusDataset(img_dir=pth_train_img,
                                                    transform=composed_train,
                                                    lbl_dir=pth_train_lbl,
                                                    mode="train")

        # bcg 0
        iver_ = _matches("_IVER_", pth_train_img)
        iwr_ = _matches("_IWR_", pth_train_img)
        ag1_ = _matches("_AG1_", pth_train_img)
        c59_ = _matches("_C59_", pth_train_img)
        cnt_ = _matches("_CONTROL_", pth_train_img)

        wts = calc_weights(iver_, iwr_, ag1_, c59_, cnt_)
        sampler = WeightedRandomSampler(wts, len(wts))

        dataset_size = len(transformed_train_data)

        train_set, val_set = torch.utils.data.random_split(
            transformed_train_data, [np.floor(0.8*dataset_size), np.floor(0.2*dataset_size)])

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=sampler, num_workers=workers)

        valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, sampler=sampler, num_workers=workers)

    return (train_loader, valid_loader)


def get_testloader(config, pth_test_img, filtered=None):
    batch_size = config.batch_size
    workers = config.num_workers

    composed_test = transforms.Compose([Normalize(), ToTensor()])

    transformed_test_data = XenopusDataset(img_dir=pth_test_img,
                                           transform=composed_test,
                                           mode="test", filtered=filtered)

    test_loader = torch.utils.data.DataLoader(
        transformed_test_data, batch_size=batch_size, num_workers=workers)

    return test_loader
