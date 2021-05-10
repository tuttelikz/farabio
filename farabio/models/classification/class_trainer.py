import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from farabio.core.convnettrainer import ConvnetTrainer
from farabio.utils.helpers import makedirs, parallel_state_dict
from farabio.utils.losses import Losses
from farabio.utils.loggers import Logger, savefig, progress_bar
from farabio.models.classification.arch import *


class ClassTrainer(ConvnetTrainer):
    """Classification trainer class. Override with custom methods here.

    Parameters
    ----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def define_data_attr(self, *args):
        self._title = self.config.title + self.config.arch
        self._train_batch_size = self.config.batch_size_train
        self._test_batch_size = self.config.batch_size_test
        self._classes = ('plane', 'car', 'bird', 'cat', 'deer',
                         'dog', 'frog', 'horse', 'ship', 'truck')

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
        self._schedule = self.config.schedule
        self._gamma = self.config.gamma
        self._num_epochs = self._num_epochs
        self._resume = self.config.resume

    def define_compute_attr(self, *args):
        self._cuda = self.config.cuda
        self._device = self.config.device
        self._num_workers = self.config.num_workers
        self._data_parallel = self.config.data_parallel

    def define_log_attr(self):
        self.best_accuracy = 0
        self._checkpoint = self.config.checkpoint

    def get_trainloader(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='/home/data/02_SSD4TB/suzy/datasets/public', train=True, download=False, transform=transform_train)
        self.train_loader = DataLoader(
            trainset, batch_size=self._train_batch_size, shuffle=True, num_workers=self._num_workers)

    def get_testloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='/home/data/02_SSD4TB/suzy/datasets/public', train=False, download=False, transform=transform_test)
        self.test_loader = DataLoader(
            testset, batch_size=self._test_batch_size, shuffle=False, num_workers=self._num_workers)

    def build_model(self):
        print("==> creating model '{}'".format(self._arch))

        class_models = {
            "densenet": DenseNet121(),
            "dpn92": DPN92(),
            "efficientnet": EfficientNetB0(),
            "googlenet": GoogLeNet(),
            "mobilenet": MobileNet(),
            "mobilenet2": MobileNetV2(),
            "preactresnet":  PreActResNet18(),
            "regnet": RegNetX_200MF(),
            "resnet": ResNet18(),
            "resnext": ResNeXt29_2x64d(),
            "senet": SENet18(),
            "shufflenet2": ShuffleNetV2(1),
            "simpledla": SimpleDLA(),
            "vgg": VGG('VGG19')
        }

        self.model = class_models[self._arch]
        print(self._arch)

        if self._cuda:
            self.model.to(self._device)
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200)

    def build_parallel_model(self):
        print("==> creating parallel model '{}'".format(self._arch))

        class_models = {
            "densenet": DenseNet121(),
            "dpn92": DPN92(),
            "efficientnet": EfficientNetB0(),
            "googlenet": GoogLeNet(),
            "mobilenet": MobileNet(),
            "mobilenet2": MobileNetV2(),
            "preactresnet":  PreActResNet18(),
            "regnet": RegNetX_200MF(),
            "resnet": ResNet18(),
            "resnext": ResNeXt29_2x64d(),
            "senet": SENet18(),
            "shufflenet2": ShuffleNetV2(1),
            "simpledla": SimpleDLA(),
            "vgg": VGG('VGG19')
        }

        self.model = class_models[self._arch]
        self.model = torch.nn.DataParallel(self.model).cuda()
        cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay)

    def on_train_epoch_start(self):
        print(f'\nEpoch: {self._epoch}')
        self.model.train()
        self.train_loss = 0
        self.correct = 0
        self.total = 0

        self.train_epoch_iter = enumerate(self.train_loader)

    def on_start_training_batch(self, args):
        self.batch_idx = args[0]
        self.inputs = args[-1][0]
        self.targets = args[-1][-1]

    def training_step(self):
        if self._cuda:
            self.inputs = self.inputs.to(self._device)
            self.targets = self.targets.to(self._device)

        self.optimizer_zero_grad()
        self.outputs = self.model(self.inputs)
        self.loss = self.criterion(self.outputs, self.targets)
        self.loss_backward()
        self.optimizer_step()

    def on_end_training_batch(self):
        self.train_loss += self.loss.item()
        _, predicted = self.outputs.max(1)
        self.total += self.targets.size(0)
        self.correct += predicted.eq(self.targets).sum().item()

        progress_bar(self.batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (self.train_loss/(self.batch_idx+1), 100.*self.correct/self.total, self.correct, self.total))

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()

    def loss_backward(self):
        self.loss.backward()

    def on_evaluate_epoch_start(self):
        self.model.eval()

        self.test_loss = 0
        self.correct = 0
        self.total = 0

        self.valid_epoch_iter = enumerate(self.test_loader)

    def on_evaluate_batch_start(self, args):
        self.batch_idx = args[0]
        self.inputs = args[-1][0]
        self.targets = args[-1][-1]

    def evaluate_batch(self, args):
        if self._cuda:
            self.inputs = self.inputs.to(self._device)
            self.targets = self.targets.to(self._device)  # async?

        # compute output
        self.outputs = self.model(self.inputs)
        self.loss = self.criterion(self.outputs, self.targets)

    def on_evaluate_batch_end(self):
        self.test_loss += self.loss.item()
        _, predicted = self.outputs.max(1)
        self.total += self.targets.size(0)
        self.correct += predicted.eq(self.targets).sum().item()

        progress_bar(self.batch_idx, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (self.test_loss/(self.batch_idx+1), 100.*self.correct/self.total, self.correct, self.total))

    def on_evaluate_epoch_end(self):
        # Save checkpoint.
        acc = 100.*self.correct/self.total
        if acc > self.best_accuracy:
            print('Saving..')
            state = {
                'net': self.model.state_dict(),
                'acc': acc,
                'epoch': self._epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            self.best_acc = acc

    def on_epoch_end(self):
        self.scheduler.step()
