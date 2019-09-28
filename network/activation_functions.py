import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    """
    
    A = 1/(1+np.exp(-Z))
    
    return A

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    return A


def relu_backward(Z):
    """
    Implement the backward propagation for a single RELU unit.

    """
    
    return np.where(Z >= 0, 1, 0)

def sigmoid_backward(Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    """
    
    s = sigmoid(Z)
    s =  s * (1-s)
    
    assert (s.shape == Z.shape)
    
    return s