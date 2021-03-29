from activfuncs import plot, x
import numpy as np

def bipolar_sigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

plot(bipolar_sigmoid)
