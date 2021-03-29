import numpy as np


class ActivFuncs:
    """Activation functions
    """
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        mask = (x > 0) * 1.0
        return mask * x

    @staticmethod
    def drelu(x):
        mask = (x > 0) * 1.0
        return mask

    @staticmethod
    def log(x):
        return 1 / (1 + np.exp(-1 * x))

    @staticmethod
    def dlog(x):
        return ActivFuncs.log(x) * (1 - ActivFuncs.log(x))

    @staticmethod
    def arctan(x):
        return np.arctan(x)

    @staticmethod
    def darctan(x):
        return 1 / (1 + x ** 2)

    @staticmethod
    def softmax(x):
        shiftx = x - np.max(x)
        exp = np.exp(shiftx)
        return exp/(exp.sum())

