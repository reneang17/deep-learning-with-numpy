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
    s = np.where(Z >= 0, 1, 0)
    
    assert (s.shape == Z.shape)
    
    return s

def sigmoid_backward(Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    """
    
    s = sigmoid(Z)
    s =  s * (1-s)
    
    assert (s.shape == Z.shape)
    
    return s

#def relu_backward(dA, cache):
#    """
#    Implement the backward propagation for a single RELU unit.

#    Arguments:
#    dA -- post-activation gradient, of any shape
#    cache -- 'Z' where we store for computing backward propagation efficiently

#    Returns:
#    dZ -- Gradient of the cost with respect to Z
#    """
#    
#    Z = cache
#    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
#    
#    # When z <= 0, you should set dz to 0 as well. 
#    dZ[Z <= 0] = 0
#    
#    assert (dZ.shape == Z.shape)
#    
#    return dZ

#def sigmoid_backward(dA, cache):
#    """
#    Implement the backward propagation for a single SIGMOID unit.

#    Arguments:
#    dA -- post-activation gradient, of any shape
#    cache -- 'Z' where we store for computing backward propagation efficiently

#    Returns:
#    dZ -- Gradient of the cost with respect to Z
#    """
#    
#    Z = cache
#    
#    s = 1/(1+np.exp(-Z))
#    dZ = dA * s * (1-s)
#    
#    assert (dZ.shape == Z.shape)
#    
#    return dZ