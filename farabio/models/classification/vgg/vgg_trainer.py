from farabio.core.convnettrainer import ConvnetTrainer
from farabio.data.biodatasets import ChestXrayDataset


class VggTrainer(ConvnetTrainer):
    """VGG-Net trainer class. Override with custom methods here.

    Parameters
    ----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def define_data_attr(self, *args):
        self._data_path = self.config.data_path
        self._batch_size_train = self.config.batch_size_train
        self._batch_size_valid = self.config.batch_size_valid
        self._batch_size_test = self.config.batch_size_test

    def define_misc_attr(self, *args):
        self._TRAIN = self.config.TRAIN
        self._TEST = self.config.TEST
        self._VAL = self.config.VAL

    def get_trainloader(self):
        self.image_datasets = {x: ChestXrayDataset(root=self._data_path, mode=x, transform=None, download=False)
                               for x in [self._TRAIN, self._VAL, self._TEST]}

        dataloaders = {TRAIN: torch.utils.data.DataLoader(image_datasets[self._TRAIN], batch_size=self._batch_size_train, shuffle=True),
                       VAL: torch.utils.data.DataLoader(image_datasets[self._VAL], batch_size=self._batch_size_valid, shuffle=True),
                       TEST: torch.utils.data.DataLoader(image_datasets[self._TEST], batch_size=self._batch_size_test, shuffle=True)}
        