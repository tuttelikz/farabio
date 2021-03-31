# https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html
# Trainer: https://github.com/PyTorchLightning/pytorch-lightning/blob/3777988502d1013508455a5fd34dc7d1a7e8e035/pytorch_lightning/trainer/trainer.py
# Module: https://github.com/PyTorchLightning/pytorch-lightning/blob/3777988502d1013508455a5fd34dc7d1a7e8e035/pytorch_lightning/core/lightning.py
# GPU accelerator: https://github.com/PyTorchLightning/pytorch-lightning/blob/d916973cdc8bffe8c8a07cd29d8be681f78ef62d/pytorch_lightning/accelerators/gpu_accelerator.py
# Training loop: https://github.com/PyTorchLightning/pytorch-lightning/blob/3777988502d1013508455a5fd34dc7d1a7e8e035/pytorch_lightning/trainer/training_loop.py
# Custom model templates
# https://www.kaggle.com/harishvutukuri/gan-pytorch-lightning
# https://pytorch-lightning.readthedocs.io/en/0.7.1/pl_examples.domain_templates.gan.html

from torch.utils.data import DataLoader
from typing import List, Optional, Union
import torch.nn as nn
import torch
import sys
from abc import ABC, abstractmethod
from farabio.utils.helpers import get_gpu_memory_map


class BaseTrainer(ABC):
    """This is the base core module for all types of trainers. \
        It inherits Python's Abstract Base Class (ABC).
    """
    @abstractmethod
    def init_attr(self, *args):
        """Override this method to initialize trainer properties

        Raises
        ------
        NotImplementedError
            If not defined
        """
        raise NotImplementedError
    
    @abstractmethod
    def build_model(self, *args):
        """Override this method to build model

        Raises
        ------
        NotImplementedError
            If not defined
        """
        raise NotImplementedError
    
    @abstractmethod
    def train(self, *args):
        """Override this method to define training loop

        Raises
        ------
        NotImplementedError
            If not defined
        """
        raise NotImplementedError
    
    def evaluate(self, *args):
        """Override this method to define evaluation loop

        Raises
        ------
        NotImplementedError
            If not defined
        """
        raise NotImplementedError
    
    @abstractmethod
    def test(self, *args):
        """Override this method to define test loop

        Raises
        ------
        NotImplementedError
            If not defined
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_trainloader(self, *args):
        """Override this method to define `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ \
        class for both train and validation datasets.

        Raises
        ------
        NotImplementedError
            If not defined
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_testloader(self, *args):
        """Override this method to define `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ \
        class for test dataset.

        Raises
        ------
        NotImplementedError
            If not defined
        """
        raise NotImplementedError
