import os
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from config import get_config
from trainer import Trainer
from farabi.data.datasets import ImageDataset
import sys

# In another terminal screen, type:
# python -m visdom.server
# The from another terminal window, python main.py

if __name__ == "__main__":
    config, unparsed = get_config()

    if config.mode == 'train':
        transforms_ = [transforms.Resize(int(config.size*1.12), Image.BICUBIC),
                       transforms.RandomCrop(config.size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        dataloader = DataLoader(ImageDataset(config.dataroot, transforms_=transforms_, unaligned=True),
                                batch_size=config.batch_size, shuffle=True, num_workers=config.n_cpu)

        trnr = Trainer(config, dataloader, mode='train')
        trnr.train()

    elif config.mode == 'test':
        # Dataset loader
        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataloader = DataLoader(ImageDataset(config.dataroot, transforms_=transforms_, mode='test'),
                                batch_size=config.batch_size, shuffle=False, num_workers=config.n_cpu)

        trnr = Trainer(config, dataloader, mode='test')
        trnr.test()

    sys.stdout.write('\n')
