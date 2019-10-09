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
    def __init__(self, n_units, input_shape=None, initializer = 'normal'):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.initializer = initializer
        
        self.W = None
        self.b = None
        
        # For debugging
        self.dW = None
        self.db = None
        

    def initialize(self):
        # Initialize the weights
        
        wshape = (self.n_units, self.input_shape[0])
        if self.initializer == 'normal':
            lim = 1 / math.sqrt(wshape[0])
            self.W  = np.random.uniform(-lim, lim, wshape)
            
        if self.initializer == 'ng':
            self.W  = np.random.randn(wshape[0], wshape[1]) / np.sqrt(wshape[1])
                       
        self.b = np.zeros(shape = (self.n_units, 1))
        
        #crosschecks
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
        A_prev = self.layer_input

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            dW = np.dot(dZ, A_prev.T) / dZ.shape[1]
            db = np.sum(dZ, axis=1, keepdims=True) / dZ.shape[1]
            
            self.dW = dW
            self.db = db

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
        self.activation_func = activation_functions[name]()
        self.trainable = True
    
    def output_shape(self):
        return self.input_shape

    def forward(self, Z, training=True):
        self.layer_input = Z
        return self.activation_func(Z)

    def backward(self, dA):
        Z = self.layer_input
        dact = self.activation_func.gradient(Z)        
        assert Z.shape == dact.shape
        
        dZ = np.multiply(dA, dact)
        assert(dZ.shape == (Z.shape))
        
        return dZ 

