from activfuncs import plot, x
import numpy as np

relu = np.vectorize(lambda x: x if x > 0 else 0, otypes=[np.float])

plot(relu, yaxis=(-0.4, 1.4))
