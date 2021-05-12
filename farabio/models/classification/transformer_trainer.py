import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm.std import tqdm
from farabio.core.convnettrainer import ConvnetTrainer
from farabio.models.classification.vit.linformer import Linformer
from farabio.models.classification.vit.efficient import ViT
from farabio.utils.losses import Losses
from farabio.utils.loggers import Logger
from farabio.data.biodatasets import RANZCRDataset


class TransformerTrainer(ConvnetTrainer):
    """Classification trainer class. Override with custom methods here.

    Parameters
    -----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def define_data_attr(self, *args):
        self._root = "/home/data/02_SSD4TB/suzy/datasets/public"
        self._batch_size = self.config.batch_size
        self._dataset = self.config.dataset

    def define_train_attr(self):
        self._lr = self.config.learning_rate
        self._gamma = self.config.gamma

    def define_model_attr(self, *args):
        self._title = self.config.title

    def define_compute_attr(self, *args):
        self._cuda = self.config.cuda
        self._device = "cuda"

    def define_misc_attr(self):
        self._seed = 42
        self.seed_everything()

    def seed_everything(self):
        random.seed(self._seed)
        os.environ["PYTHONSEED"] = str(self._seed)
        np.random.seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True

    def get_trainloader(self):
        if self._dataset == 'RANZCRDataset':
            train_dataset = RANZCRDataset(
                root=self._root, train=True, transform=None, download=False)
            self.train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=self._batch_size, shuffle=True)

    def get_testloader(self):
        if self._dataset == 'RANZCRDataset':
            valid_dataset = RANZCRDataset(
                root=self._root, train=False, transform=None, download=False)
            self.valid_loader = DataLoader(dataset=valid_dataset,
                                           batch_size=self._batch_size, shuffle=True)

    def build_model(self):
        print(f"==> creating model {self._title}")

        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )

        self.model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=11,
            transformer=efficient_transformer,
            channels=3,
        )

        if self._cuda:
            self.model.to(self._device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self._gamma)

    def on_train_epoch_start(self):
        print(f'\nEpoch: {self._epoch}')
        self.model.train()
        self.epoch_loss = 0.0
        self.epoch_accuracy = 0.0
        self._i = 0
        self.train_epoch_iter = tqdm(self.train_loader)

    def on_start_training_batch(self, args):
        self._data = args[0]
        self._label = args[-1]

    def training_step(self):
        #print(self._i)
        self._i = self._i + 1
        if self._cuda:
            self._data = self._data.to(self._device)
            self._label = self._label.to(self._device)

        self._output = self.model(self._data)
        self._label = self._label.type(torch.cuda.LongTensor)

        self.loss = self.criterion(self._output, self._label)

        self.optimizer_zero_grad()
        self.loss_backward()
        self.optimizer_step()

    def on_end_training_batch(self):
        self.acc = (self._output.argmax(dim=1) == self._label).float().mean()
        self.epoch_accuracy += self.acc / len(self.train_loader)
        self.epoch_loss += self.loss / len(self.train_loader)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()

    def loss_backward(self):
        self.loss.backward()

    def on_evaluate_epoch_start(self):
        self.model.eval()
        self._j = 0

        self.epoch_val_accuracy = 0
        self.epoch_val_loss = 0

        self.valid_epoch_iter = enumerate(self.test_loader)

    def on_evaluate_batch_start(self, args):
        self._data = args[0]
        self._label = args[-1]

    def evaluate_batch(self, args):
        self._j = self._j + 1
        if self._cuda:
            self._data = self._data.to(self._device)
            self._label = self._label.to(self._device)  # async?

        # compute output
        self.val_output = self.model(self._data)
        self.label = self.label.type(torch.cuda.LongTensor)
        self.loss = self.criterion(self.val_output, self._label)

    def on_evaluate_batch_end(self):
        self.acc = (self.val_output.argmax(dim=1) == self.label).float().mean()
        self.epoch_val_accuracy += self.acc / len(self.valid_loader)
        self.epoch_val_loss += self.val_loss / len(self.valid_loader)

    def on_epoch_end(self):
        print(
            f"Epoch: {self.epoch+1} - loss: {self.epoch_loss:.4f} - acc : {self.epoch_accuracy:.4f} - val_loss : {self.epoch_val_loss:.4f} - val_acc: {self.epoch_val_accuracy: .4f}\n"
        )
