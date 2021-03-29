from activfuncs import plot, x
import numpy as np

binary_step = np.vectorize(lambda x: 1 if x > 0 else 0, otypes=[np.float])
plot(binary_step, yaxis=(-0.4, 1.4))
