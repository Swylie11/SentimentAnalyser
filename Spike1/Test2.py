import numpy as np
import random


blue = np.arange(300)
l = np.full_like(blue, 0.001, dtype=np.double).tolist()

print(sum(l))


