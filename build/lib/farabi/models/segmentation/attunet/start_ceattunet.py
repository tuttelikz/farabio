import os
import time
import torch
from farabi.models.segmentation.Trainer_attunet import Trainer
from farabi.models.segmentation.Config_attunet import get_config
from farabi.data.dataloader import get_trainloader
from farabi.data.datasets import ImgLabelDataset, XenopusDataset
from farabi.prep.transforms import Compose, Dataraf, Datajit, ImgNormalize, ToLongTensor, ImgToTensor
from farabi.utils.helpers import calc_weights
from torchvision.transforms import transforms
from torch.utils.data import WeightedRandomSampler, DataLoader


if __name__ == "__main__":
    start_time = time.time()

    img_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/Train_200909_all/Image-sanmo-bal-aug"
    lbl_dir = "/home/DATA_Lia/data_02/DATASET_SUZY/SM_XENOPUS/DATA/Dataset/Train_200909_all/Mask-sanmo-bal-aug"

    config, unparsed = get_config()

    if config.mode == 'train':
        data_loader = get_trainloader(
            config, img_dir, lbl_dir, augment=False)
        trnr = Trainer(config, data_loader, mode='train')
        trnr.train()

    time_elapsed = time.time() - start_time

    print(
        f'Complete in {time_elapsed // 60}m {time_elapsed % 60: .2f}s')
