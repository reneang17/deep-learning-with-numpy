from __future__ import print_function, division
import math
import numpy as np
import copy

from activation_functions import Sigmoid, Relu, Softmax



class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def forward(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()


class Dense(Layer):
    """A fully-connected NN layer without activation func
    out_units: int, Number of neurons in the layer.
    input_shape: tuple, The expected input shape of the layer.
    """
    def __init__(self, out_units, input_shape=None, initializer = 'normal', lr = 0.06):
        self.layer_name ='dense'
        self.layer_input = None
        self.input_shape = input_shape
        self.out_units = out_units
        self.output_shape = (self.out_units,)
        self.trainable = True
        self.initializer = initializer
        self.lshape = None
        self.lr = lr
        self.W = None
        self.b = None


    def get_output_shape(self):
        return self.output_shape

    def initialize(self):
        """ Initialize the weights
        """
        wshape = (self.output_shape[0], self.input_shape[0])
        if self.initializer == 'normal':
            lim = np.sqrt(6) / math.sqrt(wshape[0]+wshape[1])
            self.W  = np.random.uniform(-lim, lim, wshape)

        if self.initializer == 'ng':
            self.W  = np.random.randn(wshape[0], wshape[1]) / np.sqrt(wshape[1])

        self.b = np.zeros(shape = (wshape[0], 1))
        #crosschecks
        #assert(self.W.shape == (self.out_units, self.input_shape[0]))
        #assert(self.b.shape == (self.out_units, 1))

    def forward(self, A_prev, training=True): #what is training=True for?
        self.layer_input = A_prev
        self.Z= np.dot((self.W), A_prev) + self.b
        #print(self.W.shape, self.b.shape)
        assert(self.Z.shape == (self.W.shape[0], A_prev.shape[1]))
        return self.Z

    def backward(self, dZ):
        # Save weights used during forwards pass
        W = self.W
        A_prev = self.layer_input
        norm= A_prev.shape[-1]

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            dW = np.dot(dZ, A_prev.T)/norm #(2)normalize
            db = np.sum(dZ, axis=1, keepdims=True)/norm #(2)normalize
            #self.dW = dW # No need to safe
            #self.db = db
            # Update the layer weights
            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db

        return np.dot(W.T, dZ) # return dA_prev



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
        self.layer_name = name
        self.input_shape = None
        self.activation_func = activation_functions[self.layer_name]()
        self.trainable = True

    def initialize(self):
        """ Set shape
        """
        self.output_shape = self.input_shape

    def get_output_shape(self):
        return self.output_shape

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


class Activation_SoftMax(Layer):
    """A layer that applies an activation operation to the input.
    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, input_shape = None):
        self.layer_name = 'softmax'
        self.input_shape = input_shape
        self.activation_func = Softmax()
        self.trainable = False

    def initialize(self):
        # Just to set the output shape, but not needed below
        self.output_shape = self.input_shape

    def get_output_shape(self):
        return self.output_shape

    def forward(self, Z, training=True):
        self.layer_input = Z
        return self.activation_func(Z)

    def backward(self, dA):
        Z = self.layer_input
        dact = self.activation_func.gradient(Z)
        #assert Z.shape == dact.shape

        dZ = np.sum(np.multiply(dA, dact), axis = 1)
        assert(dZ.shape == (Z.shape))

        return dZ


class Flatten(Layer):
    """A layer that flattens a 2D matrix
    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, input_shape = None):
        self.layer_name = 'flatten'
        self.input_shape = input_shape
        self.trainable = False

    def initialize(self):
        # Just to set the output shape, but not needed below
        coords_to_flatten = 1
        for i in self.input_shape:
            coords_to_flatten *=i
        self.output_shape = (coords_to_flatten,)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, Z, training=True):
        batch_size= Z.shape[-1]
        shape = (self.output_shape[0], batch_size)
        return Z.reshape(shape)

    def backward(self, dA):
        batch_size= dA.shape[-1]
        shape = self.input_shape+(batch_size,)
        return dA.reshape(shape)



class Conv2D():
    """A layer that flattens a 2D matrix
    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, input_shape= None, f = 2, n_C = 1, stride =1, padding = 0, initializer = 'normal'):
        self.layer_name= 'Conv2D'
        self.f = f
        self.initializer = initializer
        self.stride = stride
        self.pad =  padding
        self.input_shape = input_shape
        self.n_C = n_C
        self.layer_input =None

    def initialize(self):
        wshape = (self.f, self.f, self.input_shape[2], self.n_C)
        if self.initializer == 'normal':
            lim = np.sqrt(6) / math.sqrt(wshape[0]+wshape[1])
            self.W  = np.random.uniform(-lim, lim, wshape)
        self.b = np.zeros(shape = (1, 1, 1, self.n_C))
        self.get_output_shape()

    def get_output_shape(self):
        n_H = int((self.input_shape[0] - self.f + 2 * self.pad) / self.stride) + 1
        n_W = int((self.input_shape[1] - self.f + 2 * self.pad) / self.stride) + 1
        self.output_shape = (n_H, n_W, self.n_C )
        return self.output_shape



    def forward(self, A_prev):


        padding_shape = ((self.pad, self.pad), (self.pad,self.pad), (0,0), (0,0))
        A_prev_pad = np.pad(A_prev, padding_shape , 'constant', constant_values = (0,0))
        self.layer_input = A_prev_pad[...,np.newaxis,:] #inserted axis to allow for matrix multiplication

        Z = np.zeros((*self.output_shape , A_prev.shape[-1])) # Start output matrix
        Z_ishape =  Z.shape




        for h in range(0, self.output_shape[0]):           # loop over vertical axis of the output volume
            for w in range(0, self.output_shape[1]):       # loop over horizontal axis of the output volume
                # Find the corners of the current "slice" (≈4 lines)
                vert_start = h * self.stride
                vert_end = vert_start + self.f
                horiz_start = w * self.stride
                horiz_end = horiz_start + self.f

                A_slice_input = self.layer_input[vert_start: vert_end, horiz_start:horiz_end, ...]
                arr = (np.multiply(A_slice_input, self.W[..., np.newaxis] )) +self.b[..., np.newaxis]
                Z[h, w, ...] = arr.sum(axis=tuple(range(arr.ndim - 2))) # sums everything except the last two dims

        assert Z_ishape == Z.shape, 'Dimensions in convolution do not match'

        return Z

    def backward(self, dZ):
        pass
