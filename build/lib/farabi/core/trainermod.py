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
    """Base of trainer module (inherits ABC)

    """

    def __init__(self, *args):
        """[summary]
        """
        super().__init__()
        config = args[0]

        # common configurations for every model

        self._save_epoch = config.save_epoch
        self._num_epochs = config.num_epochs
        self._mode = config.mode
        self._start_epoch = config.start_epoch
        self._data_parallel = config.data_parallel
        self.init_attributes()

        if self._data_parallel is True:
            self._device = "cuda:1"
        elif self.data_parallel is False:
            gpu_memory = get_gpu_memory_map()
            gpu_id = min(gpu_memory, key=gpu_memory.get)
            self._device = "cuda:"+str(gpu_id)

        self.build_model()

        if self._data_parallel is False:
            self.model.to(self.device)
        elif self.data_parallel is True:
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

        self.get_dataloader()

        if hasattr(config, "batch_size"):
            self._batch_size = config.batch_size
        else:
            self._batch_size = get_dataloader[0].batch_size

        self._epoch_train_losses = []
        self._epoch_valid_losses = []
        self.init_optimizers(config.learning_rate)

        self.model_load_dir = config.model_load_dir

    @property
    def batch_size(self):
        """
        Batch size

        :return: Batch size
        :rtype: int
        """
        return self._batch_size

    @property
    def save_epoch(self):
        """
        Save after epoch

        :return: Save after epoch
        :rtype: int
        """
        return self._save_epoch

    @property
    def num_epochs(self):
        """
        Number of training epochs

        :return: Number of training epochs
        :rtype: int
        """
        return self._num_epochs

    @property
    def mode(self):
        """
        Trainer mode

        :return: Mode (Train/Test)
        :rtype: string
        """
        return self._mode

    @property
    def data_parallel(self):
        """
        Data parallel in PyTorch

        :return: Data parallel in PyTorch
        :rtype: int
        """
        return self._data_parallel

    @property
    def start_epoch(self):
        """
        Starting epoch

        :return: Starting epoch
        :rtype: int
        """
        return self._start_epoch

    @abstractmethod
    def init_attributes(self, *args):
        """Abstract method that initializes object attributes
        """
        pass

    @abstractmethod
    def init_optimizers(self, *args):
        """Abstract method that initializes optimizers
        """
        pass

    @abstractmethod
    def build_model(self, *args):
        """Abstract method that builds model
        """
        pass

    def train(self):
        """Training loop with hooks

        Lifecycle:
        ------------
        on_train_start()
        start_logger()
            on_train_epoch_start()
                start_training_batch()
                optimizer_zero_grad()
                training_step()
                backward()
                on_end_training_batch()
            on_train_epoch_end()
            evaluate()
        on_train_end()
        """
        # hook to do on tran instart
        self.on_train_start()
        self.start_logger()

        for epoch in range(self.start_epoch, self.num_epochs):
            self.on_train_epoch_start()

            for batch_idx, batch in enumerate(self.train_loader):
                self.start_training_batch()
                self.optimizer_zero_grad()
                self.loss = self.training_step(batch, batch_idx)
                self.batch_training_loss += self.loss.item()

                self.backward()
                self.optimizer_step()

                self.on_end_training_batch(
                    epoch, batch_idx, self.loss.item(), len(self.train_loader))

            self.on_train_epoch_end(
                epoch, self.batch_training_loss, len(self.train_loader))

            self.evaluate(epoch)

        self.on_train_end()

    def evaluate(self, epoch):
        """Evaluation loop with hooks

        Parameters
        ----------
        epoch : int
            Current epoch

        Lifecycle:
        ------------
        on_evaluate_start()
            evaluate_step()
        on_evaluate_end()
        """

        self.on_evaluate_start()

        batch_validation_loss = 0
        for batch_idx, batch in enumerate(self.valid_loader):
            loss = self.evaluate_step(batch, batch_idx)
            batch_validation_loss += loss

        self.on_evaluate_end(batch_validation_loss,
                             len(self.valid_loader), epoch)

    def test(self):
        """Test loop with hooks

        Lifecycle:
        ------------
        on_test_start()
            on_start_test_batch()
            test_step()
            on_end_test_batch()
        on_test_end()
        """
        self.load_model()
        self.on_test_start()

        for batch_idx, batch in enumerate(self.test_loader):
            self.on_start_test_batch()
            self.test_step(batch, batch_idx)
            self.on_end_test_batch()
        self.on_test_end()

    def get_dataloader(self):
        """Retreives torch.utils.data.DataLoader object
        """
        pass

    # Training loop related
    def on_train_start(self):
        """Hook: On start of training loop
        """

        self.batch_training_loss = 0

    def start_logger(self, *args):
        """Hook: Starts logger
        """
        pass

    def on_train_epoch_start(self):
        """Hook: On epoch start
        """
        self.batch_training_loss = 0
        self.model.train()

    def start_training_batch(self):
        """Hook: On training batch start
        """
        pass

    def optimizer_zero_grad(self):
        """Zero grad
        """
        self.optimizer.zero_grad()

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Hook: Training action (Put training here)

        Parameters
        ----------
        batch : tuple
            receives batch
        batch_idx : int
            index of batch
        """

        pass

    def backward(self):
        """Hook: sends backward
        """
        self.loss.backward()

    def optimizer_step(self):
        """Optimizer step
        """
        self.optimizer.step()

    def on_end_training_batch(self, *args):
        """Hook: On end of training batch
        """
        pass

    def on_train_epoch_end(self, *args):
        """Hook: On end of training epoch
        """
        pass

    def on_train_end(self):
        """Hook: On end of training
        """
        pass

    def stop_train(self, *args):
        """On end of training
        """
        sys.exit()

    # Evaluation loop related

    def on_evaluate_start(self):
        """Hook: on evaluation start
        """
        self.model.eval()

    @abstractmethod
    def evaluate_step(self):
        """Evaluation action (Put evaluation here)
        """
        pass

    def on_evaluate_end(self, *args):
        """Hook: on evaluation end
        """
        pass

    # Test loop related
    def on_test_start(self, *args):
        """Hook: on test start
        """
        self.model.eval()

    def on_start_test_batch(self, *args):
        """Hook: on test batch start
        """
        pass

    @abstractmethod
    def test_step(self, *args):
        """Test action (Put test here)
        """
        pass

    def on_end_test_batch(self, *args):
        """Hook: on end of batch test
        """
        pass

    def on_test_end(self, *args):
        """Hook: on end test
        """
        pass

    # Misc related
    def load_model(self, *args):
        """Hook: load model
        """
        self.model.load_state_dict(torch.load(self.model_load_dir))

    def save_model(self, *args):
        """Hook: saves model
        """
        pass
