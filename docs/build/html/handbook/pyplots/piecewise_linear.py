from activfuncs import plot, x
import numpy as np

piecewise_linear = np.vectorize(
    lambda x: 1 if x > 3 else 0 if x < -3 else 1/6*x+1/2, otypes=[np.float])
plot(piecewise_linear, yaxis=(-0.4, 1.4))
