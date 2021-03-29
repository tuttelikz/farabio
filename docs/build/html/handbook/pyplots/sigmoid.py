from activfuncs import plot, x
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plot(sigmoid, yaxis=(-0.4, 1.4))
