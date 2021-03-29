import os
import time
from farabi.data.dataloader import get_trainloader, get_testloader
from trainer import Trainer
from config import get_config
from farabi.utils.helpers import x, makedirs
from farabi.data.datasets import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder
from torch.utils.data import DataLoader
from srgan import Generator, Discriminator


if __name__ == "__main__":
    start_time = time.time()

    config, unparsed = get_config()

    if config.mode == "train":
        train_set = TrainDatasetFromFolder(
            '/data/02_SSD4TB/suzy/datasets/public/div2k/DIV2K_train_HR', crop_size=config.crop_size, upscale_factor=config.upscale_factor)
        val_set = ValDatasetFromFolder(
            '/data/02_SSD4TB/suzy/datasets/public/div2k/DIV2K_valid_HR', upscale_factor=config.upscale_factor)
        train_loader = DataLoader(
            dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
        val_loader = DataLoader(dataset=val_set, num_workers=4,
                                batch_size=1, shuffle=False)
        trnr = Trainer(config, (train_loader, val_loader), mode=config.mode)
        trnr.train()

    if config.mode == "test":
        # test_set = TestDatasetFromFolder(
        #     '/home/DATA_Lia/data_02/DATASET_EXT/SuperResolution/lowres_test', upscale_factor=config.upscale_factor)
        test_set = TestDatasetFromFolder(
            '/home/suzy/images', upscale_factor=config.upscale_factor)
        test_loader = DataLoader(
            dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

        trnr = Trainer(config, test_loader, mode=config.mode)
        trnr.test()

    time_elapsed = time.time() - start_time

    print(
        f'Complete in {time_elapsed // 60}m {time_elapsed % 60: .2f}s')
