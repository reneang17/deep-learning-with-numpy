from __future__ import print_function, division
import math
import numpy as np
import copy

from activation_functions import Sigmoid, Relu



class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def forward(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()


class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        
        self.W = None
        self.b = None
        

    def initialize(self):
        # Initialize the weights
        np.random.seed(3)
        lim = 1 / math.sqrt(self.input_shape[0])
        
        self.W  = np.random.uniform(-lim, lim, 
                  (self.n_units, self.input_shape[0]))
        self.b = np.zeros(shape=(self.n_units, 1))
        
        assert(self.W.shape == (self.n_units, self.input_shape[0]))
        assert(self.b.shape == (self.n_units, 1))

    def output_shape(self):
        return (self.n_units,)
        
    
    def forward(self, A_prev, training=True): #what is training=True for?
        
        self.layer_input = A_prev
        self.Z= np.dot((self.W), A_prev) + self.b
        
        assert(self.Z.shape == (self.W.shape[0], A_prev.shape[1]))
        return self.Z
    
    def backward(self, dZ):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            dW = np.dot(dZ, A_prev.T) / dZ.shape[1]
            db = np.sum(dZ, axis=1, keepdims=True) / dZ.shape[1]

            # Update the layer weights
            learning_rate = 0.0075
            self.W = self.W - learning_rate * dW 
            self.b = self.b - learning_rate * db

        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev

        
        
activation_functions = {
    'sigmoid': Sigmoid,
    'relu': Relu    
}

class Activation(Layer):
    """A layer that applies an activation operation to the input.
    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.act_func = activation_functions[name]()
        self.trainable = True
    
    def output_shape(self):
        return self.input_shape

    def forward(self, Z, training=True):
        self.layer_input = Z
        return self.act_func(Z)

    def backward(self, dA):
        dact = self.activation_func.gradient(self.layer_input)
        dZ = dA * dact
        assert(dZ.shape == dA.shape)
        assert(dZ.shape == dact.shape)
        return dZ 

