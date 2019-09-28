from __future__ import print_function, division
import math
import numpy as np
import copy

from activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU
from activation_functions import TanH, ELU, SELU, Softmax



class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def forward_linear(self, X, training):
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
        self.dW = None
        self.db = None        
        
        self.A = None
        self.Z = None
        self.dA = None
        self.dZ = None
        
    def output_shape(self):
        return (self.n_units,)

    def initialize(self):
        # Initialize the weights
        np.random.seed(3)
        lim = 1 / math.sqrt(self.input_shape[0])
        
        self.W  = np.random.uniform(-lim, lim, 
                  (self.n_units, self.input_shape[0]))
        self.b = np.zeros(shape=(self.n_units, 1))
        
        assert(self.W.shape == (self.n_units, self.input_shape[0]))
        assert(self.b.shape == (self.n_units, 1))
        
    
    def forward_linear(self, A_prev, training=True): #what is training=True for?
        
        self.layer_input = A_prev
        self.Z= np.dot((self.W), A_prev) + self.b
        
        assert(self.Z.shape == (self.W.shape[0], A_prev.shape[1]))
        return self.Z
        
        
