from activfuncs import plot, x
import numpy as np

bipolar = np.vectorize(lambda x: 1 if x > 0 else -1, otypes=[np.float])
plot(bipolar)
