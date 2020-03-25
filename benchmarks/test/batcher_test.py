from batcher import batcher
import numpy as np

X= np.random.rand(3,10)

test = batcher(X, batch_size = 1)

X
next(test)
X[:,0]
