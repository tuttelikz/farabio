from activfuncs import plot, x
import numpy as np

def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) -1

plot(tanh)
