from activfuncs import plot, x
import numpy as np

leaky_relu = np.vectorize(lambda x: max(0.1 * x, x), otypes=[np.float])

plot(leaky_relu)
