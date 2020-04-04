import numpy as np

def batcher(X, y=None, batch_size=64):
    """ batch generator """
    n_samples = X.shape[-1]
    for i in np.arange(0, n_samples, batch_size):
        start, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[..., start:end], y[..., start:end]
        else:
            yield X[..., start:end]



class Batcher:
    def __init__(self, X, y=None, batch_size=64):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        return batcher(self.X, self.y, self.batch_size)
