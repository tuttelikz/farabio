import torch
import sys
from farabio.core.basetrainer import BaseTrainer

__all__ = ['ConvnetTrainer']


class ConvnetTrainer(BaseTrainer):
    """ConvnetTrainer is main trainer class for every ConvNet related
    architectures.

    Parameters
    ----------
    BaseTrainer : ABC
        Inherits BaseTrainer class
    """

    def __init__(self, config):
        """Initializes trainer object
        """
        super().__init__()
        self.config = config
        self.default_attr()
        self.init_attr()
        self.get_trainloader()
        self.get_testloader()

        if self._data_parallel:
            self.build_parallel_model()
        elif not self._data_parallel:
            self.build_model()

    ##########################
    # Definition of attributes
    ##########################
    def default_attr(self, *args):
        self._num_epochs = 10
        self._mode = 'train'
        self._save_epoch = 1
        self._start_epoch = 1
        self._has_eval = True
        self._eval_interval = 1
        self._train_loader = None
        self._valid_loader = None
        self._test_loader = None
        self._model = None
        self._model_path = None
        self._train_epoch_iter = None
        self._valid_epoch_iter = None
        self._test_loop_iter = None
        self._use_tqdm = False
        self._data_parallel = None
        self._next_loop = False
        self._epoch = 1
        self._backbone = False

    def init_attr(self, *args):
        """Abstract method that initializes object attributes
        """
        self.define_data_attr()
        self.define_model_attr()
        self.define_train_attr()
        self.define_test_attr()
        self.define_log_attr()
        self.define_compute_attr()
        self.define_misc_attr()

    def define_data_attr(self, *args):
        """Define data related attributes here
        """
        pass

    def define_model_attr(self, *args):
        """Define model related attributes here
        """
        pass

    def define_train_attr(self, *args):
        """Define training related attributes here
        """
        pass

    def define_test_attr(self, *args):
        """Define training related attributes here
        """
        pass

    def define_log_attr(self, *args):
        """Define log related attributes here
        """
        pass

    def define_compute_attr(self, *args):
        """Define compute related attributes here
        """
        pass

    def define_misc_attr(self, *args):
        """Define miscellaneous attributes here
        """
        pass

    ##########################
    # Building model
    ##########################
    def build_model(self, *args):
        """Abstract method that builds model
        """
        pass

    def build_parallel_model(self, *args):
        """Abstract method that builds multi-GPU model in parallel
        """
        pass

    ##########################
    # Dataloaders
    ##########################
    def get_trainloader(self, *args):
        """Hook: Retreives training set of torch.utils.data.DataLoader class
        """
        pass

    def get_testloader(self, *args):
        """Hook: Retreives test set of torch.utils.data.DataLoader class
        """
        pass

    ##########################
    # Training related
    ##########################
    def train(self):
        """Training loop with hooksa
        """
        self.on_train_start()
        self.start_logger()
        self.train_loop()
        self.on_train_end()

    def train_loop(self):
        """Hook: training loop
        """
        for self._epoch in range(self._start_epoch, self._num_epochs+1):
            self.train_epoch()

    def train_epoch(self):
        """Hook: epoch of training loop
        """
        self.on_train_epoch_start()

        for train_epoch_var in self._train_epoch_iter:
            self.train_batch(train_epoch_var)

        self.on_train_epoch_end()

        if self._has_eval:
            if self._epoch % self._eval_interval == 0:
                self.evaluate_epoch()

        self.on_epoch_end()

    def train_batch(self, args):
        """Hook: batch of training loop
        """
        self.on_start_training_batch(args)

        self.training_step()

        self.on_end_training_batch()

    def on_train_start(self):
        """Hook: On start of training loop
        """
        self._batch_training_loss = 0

    def start_logger(self, *args):
        """Hook: Starts logger
        """
        pass

    def on_train_epoch_start(self):
        """Hook: On epoch start
        """
        self._batch_training_loss = 0
        self._model.train()

    def on_start_training_batch(self, *args):
        """Hook: On training batch start
        """
        pass

    def optimizer_zero_grad(self, *args):
        """Hook: Zero gradients of optimizer
        """

    def training_step(self, *args):
        """Hook: During training batch
        """
        pass

    def loss_backward(self, *args):
        """Hook: Loss back-propagation
        """

    def optimizer_step(self):
        """Hook: Optimizer step
        """
        pass

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

    def on_epoch_end(self, *args):
        """Hook: on epoch end
        """
        pass

    def stop_train(self, *args):
        """On end of training
        """
        sys.exit()

    ##########################
    # Evaluation loop related
    ##########################
    def evaluate_epoch(self):
        """Hook: epoch of evaluation loop

        Parameters
        ----------
        epoch : int
            Current epoch
        """

        with torch.no_grad():
            self.on_evaluate_epoch_start()

            for valid_epoch_var in self._valid_epoch_iter:
                self.on_evaluate_batch_start(valid_epoch_var)
                if self._next_loop:
                    self._next_loop = False
                    continue
                self.evaluate_batch(valid_epoch_var)
                self.on_evaluate_batch_end()
            self.on_evaluate_epoch_end()

    def evaluate_batch(self, *args):
        """Hook: batch of evaluation loop
        """
        pass

    def on_evaluate_start(self, *args):
        """Hook: on evaluation end
        """
        pass

    def on_evaluate_epoch_start(self):
        """Hook: on evaluation start
        """
        raise NotImplementedError

    def on_evaluate_batch_start(self, *args):
        pass

    def on_evaluate_batch_end(self):
        """Hook: On evaluate batch end
        """
        pass

    def on_evaluate_epoch_end(self, *args):
        pass

    def on_evaluate_end(self, *args):
        """Hook: on evaluation end
        """
        pass

    ##########################
    # Test loop related
    ##########################
    def test(self):
        """Hook: Test lifecycle
        """

        if self._data_parallel:
            self.load_parallel_model()
        elif not self._data_parallel:
            self.load_model()
        self.on_test_start()
        self.test_loop()
        self.on_test_end()

    def test_loop(self):
        """Hook: test loop
        """
        for test_loop_var in self._test_loop_iter:
            self.on_start_test_batch()
            self.test_step(test_loop_var)
            self.on_end_test_batch()

    def on_test_start(self, *args):
        """Hook: on test start
        """
        pass

    def on_start_test_batch(self, *args):
        """Hook: on test batch start
        """
        pass

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

    ##########################
    # Handle models
    ##########################
    def load_model(self, *args):
        """Hook: load model
        """
        pass

    def load_parallel_model(self, *args):
        """Hook: load parallel model
        """
        pass

    def save_model(self, *args):
        """Hook: saves model
        """
        pass

    def save_parallel_model(self, *args):
        """Hook: saves parallel model
        """
        pass

    ##########################
    # Miscellaneous
    ##########################
    def exit_trainer(self, *args):
        """Exits trainer
        """
        sys.exit()
