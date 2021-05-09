import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensor
from farabio.core.convnettrainer import ConvnetTrainer
from farabio.models.segmentation.unet.unet import Unet
from farabio.utils.regul import EarlyStopping
from farabio.utils.losses import Losses
from farabio.utils.tensorboard import TensorBoard
from farabio.utils.helpers import makedirs, parallel_state_dict
import skimage
import torchvision
from skimage import io, transform, img_as_ubyte
from skimage.io import imsave
from torchsummary import summary
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from farabio.data.biodatasets import DSB18Dataset
import farabio.models.classification.arch as models


class AlexTrainer(ConvnetTrainer):
    """U-Net trainer class. Override with custom methods here.

    Parameters
    ----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def define_data_attr(self, *args):
        self._train_batch_size = self.config.batch_size_train
        self._test_batch_size = self.config.batch_size_test
    
    def define_model_attr(self, *args):
        self._arch = self.config.arch
        self._cardinality = self.config.cardinality
        self._num_classes = self.config.num_classes
        self._depth = self.config.depth
        self._widen_factor = self.config.widen_factor
        self._drop_rate = self.config.dropout
        self._growth_rate = self.config.growth_rate
        self._compression_rate = self.config.compression_rate
        self._block_name = self.config.block_name

    def define_train_attr(self):
        self._lr = self.config.learning_rate
        self._momentum = self.config.momentum
        self._weight_decay = self.config.weight_decay

    def define_compute_attr(self, *args):
        self._cuda = self.config.cuda
        self._device = self.config.device
        self._num_workers = self.config.num_workers
        self._data_parallel = self.config.data_parallel

    def get_trainloader(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='/home/data/02_SSD4TB/suzy/datasets/public', train=True, download=False, transform=transform_train)
        self.train_loader = DataLoader(trainset, batch_size=self._train_batch_size, shuffle=True, num_workers=self._num_workers)

    def get_testloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='/home/data/02_SSD4TB/suzy/datasets/public', train=False, download=False, transform=transform_test)
        self.test_loader = DataLoader(testset, batch_size=self._test_batch_size, shuffle=False, num_workers=self._num_workers)

    def build_model(self):
        # Model
        print("==> creating model '{}'".format(self._arch))
        if self._arch.startswith('resnext'):
            self.model = models.__dict__[self._arch](
                        cardinality=self._cardinality,
                        num_classes=self._num_classes,
                        depth=self._depth,
                        widen_factor=self._widen_factor,
                        dropRate=self._drop,
                    )
        elif self._arch.startswith('densenet'):
            self.model = models.__dict__[self._arch](
                        num_classes=self._num_classes,
                        depth=self._depth,
                        growthRate=self._growth_rate,
                        compressionRate=self._compression_rate,
                        dropRate=self._drop,
                    )
        elif self._arch.startswith('wrn'):
            self.model = models.__dict__[self._arch](
                        num_classes=self._num_classes,
                        depth=self._depth,
                        widen_factor=self._widen_factor,
                        dropRate=self._drop_rate,
                    )
        elif self._arch.endswith('resnet'):
            self.model = models.__dict__[self._arch](
                        num_classes=self._num_classes,
                        depth=self._depth,
                        block_name=self._block_name,
                    )
        else:
            self.model = models.__dict__[self._arch](num_classes=self._num_classes)

        if self._cuda:
            self.model.to(self._device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay)

    def build_parallel_model(self):
        # Model
        print("==> creating model '{}'".format(self._arch))
        if self._arch.startswith('resnext'):
            self.model = models.__dict__[self._arch](
                        cardinality=self._cardinality,
                        num_classes=self._num_classes,
                        depth=self._depth,
                        widen_factor=self._widen_factor,
                        dropRate=self._drop,
                    )
        elif self._arch.startswith('densenet'):
            self.model = models.__dict__[self._arch](
                        num_classes=self._num_classes,
                        depth=self._depth,
                        growthRate=self._growth_rate,
                        compressionRate=self._compression_rate,
                        dropRate=self._drop,
                    )
        elif self._arch.startswith('wrn'):
            self.model = models.__dict__[self._arch](
                        num_classes=self._num_classes,
                        depth=self._depth,
                        widen_factor=self._widen_factor,
                        dropRate=self._drop_rate,
                    )
        elif self._arch.endswith('resnet'):
            self.model = models.__dict__[self._arch](
                        num_classes=self._num_classes,
                        depth=self._depth,
                        block_name=self._block_name,
                    )
        else:
            self.model = models.__dict__[self._arch](num_classes=self._num_classes)

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay)