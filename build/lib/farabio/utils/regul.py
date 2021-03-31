import numpy as np
import torch
import os


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Attributes
    ----------
    patience : int, optional
        how long to wait after last time validation loss improved, by default 7
    verbose : bool, optional
        if True, prints a message for each validation loss improvement, by default False
    best_score : float
        stores best score
    early_stop : bool
        flag for early stopping
    val_loss_min : float
        minimum validation loss
    delta : float, optional
        minimum change in the monitored quantity to qualify as an improvement, by default 0
    trace_func : function, optional
        trace print function, by default print

    Methods
    -------
    __init__(self, patience=7, verbose=False, delta=0, trace_func=print)
        Constructor for EarlyStopping class
    __call__(self, val_loss, model, path='./')
        Passes class function
    save_checkpoint(self, val_loss, model)
        Method to save checkpoints
    """

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """Constructor for EarlyStopping

        Parameters
        ----------
        patience : int, optional
            how long to wait after last time validation loss improved, by default 7
        verbose : bool, optional
            if True, prints a message for each validation loss improvement, by default False
        delta : float, optional
            minimum change in the monitored quantity to qualify as an improvement, by default 0
        trace_func : function, optional
            trace print function, by default print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model, path='./'):
        """[summary]

        Parameters
        ----------
        val_loss : float
            validation loss
        model : nn.Module
            deep learning model
        path : str, optional
            path to save checkpoints, by default './'
        """
        score = -val_loss

        self.path = os.path.join(path, "checkpoint.pt")

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease.
        """

        # if self.verbose:
        #   self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) >
                0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
