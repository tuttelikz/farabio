from farabio.core.convnettrainer import ConvnetTrainer
from farabio.models.classification import *

class_models = {
    'alexnet': alexnet(),
    'googlenet': googlenet(),
    'mobilenetv2': mobilenet_v2(),
    'mobilenetv3-l': mobilenet_v3_large(),
    'mobilenetv3-s': mobilenet_v3_small(),
    'resnet18': resnet18(), 
    'resnet34': resnet34(),
    'resnet50': resnet50(),
    'resnet101': resnet101(),
    'resnet152': resnet152(),
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5(),
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0(),
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5(),
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0(),
    'squeezenet': squeezenet(),
    'vgg11': vgg11(),
    'vgg11_bn': vgg11_bn(),
    'vgg13': vgg13(),
    'vgg13_bn': vgg13_bn(),
    'vgg16': vgg16(),
    'vgg16_bn': vgg16_bn(),
    'vgg19': vgg19(),
    'vgg19_bn': vgg19_bn()
}

class ClassTrainer(ConvnetTrainer):
    """Classification trainer class. Override with custom methods here.

    Parameters
    -----------
    ConvnetTrainer : BaseTrainer
        Inherits ConvnetTrainer class
    """

    def define_data_attr(self, *args):
        self._arch = self.config.arch
        self._num_classes = self.config.num_classes
        self._classes = self.config.class_names
        self._train_batch_size = self.config.batch_size_train
        self._test_batch_size = self.config.batch_size_test

    def define_train_attr(self):
        self._num_epochs = self._num_epochs
        self._lr = self.config.learning_rate
        self._momentum = self.config.momentum
        self._weight_decay = self.config.weight_decay
        self._schedule = self.config.schedule
        self._gamma = self.config.gamma

    def define_compute_attr(self, *args):
        self._num_workers = self.config.num_workers
        self._cuda = self.config.cuda
        self._device = self.config.device
        self._data_parallel = self.config.data_parallel
    
    def define_log_attr(self):
        self._best_accuracy = 0
        self._checkpoint = self.config.checkpoint

    def get_trainloader(self):
        print("train_loader")

    def get_testloader(self):
        print("test_loader")
    
    def build_model(self):
        print("creating model")

        self._model = class_models[self._arch]
        if self._cuda:
            