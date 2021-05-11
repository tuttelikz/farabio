import os
import time
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
from farabio.utils.loggers import Logger, savefig
from farabio.utils.metrics import accuracy
from farabio.utils.meters import AverageMeter
import farabio.models.classification.conv as models
from progress.bar import Bar


class AlexTrainer(ConvnetTrainer):
    """U-Net trainer class. Override with custom methods here.

    Parameters
    ----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def define_data_attr(self, *args):
        self._title = self.config.title + self.config.arch
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
        self._schedule = self.config.schedule
        self._gamma = self.config.gamma
        self._num_epochs = self._num_epochs
    
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

    def start_logger(self):
       self.logger = Logger(os.path.join(self._checkpoint, 'log.txt'), title=self._title)
       self.logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    def on_train_epoch_start(self):
        self.model.train()
        
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.end = time.time()

        self.bar = Bar('Processing', max=len(self.train_loader))

        self.adjust_learning_rate()
        self.train_epoch_iter = enumerate(self.train_loader)

        print(f'\nEpoch: [{self._epoch + 1} | {self._num_epochs}] LR: {self._lr}')
    
    def on_start_training_batch(self, args):
        self.batch_idx = args[0]
        self.inputs = args[-1][0]
        self.targets = args[-1][-1]
        self.data_time.update(time.time() - self.end)
    
    def training_step(self):
        if self._cuda:
            self.inputs = self.inputs.to(self._device)
            self.targets = self.targets.to(self._device) #async?

        self.inputs, self.targets = torch.autograd.Variable(self.inputs), torch.autograd.Variable(self.targets)

        self.outputs = self.model(self.inputs)
        self.loss = self.criterion(self.outputs, self.targets)
        prec1, prec5 = accuracy(self.outputs.data, self.targets.data, topk=(1, 5))

        self.losses.update(self.loss.data[0], self.inputs.size(0))
        self.top1.update(prec1[0], self.inputs.size(0))
        self.top5.update(prec5[0], self.inputs.size(0))

        self.optimizer_zero_grad()
        self.loss_backward()
        self.optimizer_step()

    def on_end_training_batch(self):
        # measure elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

        # plot progress
        self.bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=self.batch_idx + 1,
            size=len(self.train_loader),
            data=self.data_time.avg,
            bt=self.batch_time.avg,
            total=self.bar.elapsed_td,
            eta=self.bar.eta_td,
            loss=self.losses.avg,
            top1=self.top1.avg,
            top5=self.top5.avg,
            )

        self.bar.next()

    def on_train_epoch_end(self):
        self.bar.finish()
        self.train_loss = self.losses.avg
        self.train_accuracy = self.top1.avg
        
    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()
    
    def optimizer_step(self):
        self.optimizer.step()
    
    def loss_backward(self):
        self.loss.backward()

    def on_evaluate_epoch_start(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        
        self.model.eval()
        
        self.end = time.time()
        self.valid_epoch_iter = enumerate(self.test_loader)

        self.bar = Bar('Processing', max=len(self.test_loader))

    def on_evaluate_batch_start(self, args):
        self.data_time.update(time.time() - self.end)
        
        self.batch_idx = args[0]
        self.inputs = args[-1][0]
        self.targets = args[-1][-1]

    def evaluate_batch(self, args):
        if self._cuda:
            self.inputs = self.inputs.to(self._device)
            self.targets = self.targets.to(self._device) #async?

        self.inputs = torch.autograd.Variable(self.inputs, volatile=True)
        self.targets = torch.autograd.Variable(self.targets)

        # compute output
        self.outputs = self.model(self.inputs)
        self.loss = self.criterion(self.outputs, self.targets)

        # measure accuracy and record loss
        self.prec1, self.prec5 = accuracy(self.outputs.data, self.targets.data, topk=(1, 5))
        
    def on_evaluate_batch_end(self):
        self.losses.update(self.loss.data[0], self.inputs.size(0))
        self.top1.update(self.prec1[0], self.inputs.size(0))
        self.top5.update(self.prec5[0], self.inputs.size(0))
        # measure elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()
        # plot progress
        self.bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=self.batch_idx + 1,
                    size=len(self.test_loader),
                    data=self.data_time.avg,
                    bt=self.batch_time.avg,
                    total=self.bar.elapsed_td,
                    eta=self.bar.eta_td,
                    loss=self.losses.avg,
                    top1=self.top1.avg,
                    top5=self.top5.avg,
                    )
        self.bar.next()
    
    def on_evaluate_epoch_end(self):
        self.bar.finish()
        self.test_loss = self.losses.avg
        self.test_accuracy = self.top1.avg

    def on_epoch_end(self):
        # append logger file
        self.logger.append([state['lr'], self.train_loss, self.test_loss, self.train_accuracy, self.test_accuracy])

        # save model
        is_best = self.test_accuracy > self.best_accuracy
        self.best_accuracy = max(self.test_accuracy, self.best_accuracy)
        self.save_checkpoint({
                'epoch': self._epoch + 1,
                'state_dict': self.model.state_dict(),
                'acc': self.test_accuracy,
                'best_acc': self.best_accuracy,
                'optimizer' : self.optimizer.state_dict(),
            }, is_best, checkpoint=self.config.checkpoint)

    def on_train_end(self):
        self.logger.close()
        self.logger.plot()
        savefig(os.path.join(self._checkpoint, 'log.eps'))

        print('Best acc:')
        print(self.best_accuracy)

    def adjust_learning_rate(self):
        if self._epoch in self._schedule:
            self._lr *= self._gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._lr

    def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))