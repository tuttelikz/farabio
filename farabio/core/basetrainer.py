from abc import ABC, abstractmethod

__all__ = ['BaseTrainer']


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
