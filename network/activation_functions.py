import numpy as np

class Sigmoid():
    """
    Fordward/backward sigmoid propagation
    in numpy
    """

    def __call__(self, Z):
        return 1 / (1 + np.exp(-Z))

    def gradient(self, A):
        return self.__call__(A) * (1 - self.__call__(A))

class Relu():
    """
    Fordward/backward ReLU propagation
    in numpy
    """
    def __call__(self, Z):
        return np.where(Z >= 0, Z, 0)

    def gradient(self, Z):
        return np.where(Z >= 0, 1, 0)


class Softmax():
    """
    Fordward/backward Softmax propagation
    in numpy
    """
    def __call__(self, Z):
        e_Z = np.exp(Z)
        out = e_Z / np.sum(e_Z, axis=1, keepdims=True)
        #assert Z.shape[0] - 10**(-13) <np.sum(out) < Z.shape[0] + 10**(-13), print(Z, e_Z )
        assert Z.shape == out.shape
        return out

    def gradient(self, Z):
        p = self.__call__(Z)
        grad = - p[:, :, np.newaxis] *  p[:, np.newaxis, :]
        diag = np.arange(p.shape[-1])
        grad[:, diag, diag]  = p * (1-p)

        return grad
