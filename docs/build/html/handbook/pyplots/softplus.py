from activfuncs import plot, x
import numpy as np

def softplus(x):
    return np.log(1+np.exp(x))

plot(softplus, yaxis=(-0.4, 1.4))
