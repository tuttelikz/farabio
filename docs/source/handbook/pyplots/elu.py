from activfuncs import plot, x
import numpy as np

elu = np.vectorize(lambda x: x if x > 0 else 0.5 * (np.exp(x) - 1), otypes=[np.float])

plot(elu)
