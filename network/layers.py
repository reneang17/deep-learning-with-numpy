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
    """A fully-connected layer
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
        """ Initialize the weights. Unchanged
        """
        wshape = (self.input_shape[0], self.output_shape[0])

        if self.initializer == 'normal':
            lim = np.sqrt(6) / math.sqrt(wshape[0]+wshape[1])
            self.W  = np.random.uniform(-lim, lim, wshape)

        if self.initializer == 'ng':
            self.W  = np.random.randn(wshape[0], wshape[1]) / np.sqrt(wshape[0])

        self.b = np.zeros(shape = (1, wshape[1]))
        assert self.W.shape == (wshape[0], self.out_units)
        assert (self.b.shape == (1,self.out_units))

    def forward(self, A_prev, training=True): #what is training=True for?
        self.layer_input = A_prev
        self.Z= np.dot(A_prev, self.W) + self.b
        assert self.Z.shape == (A_prev.shape[0], self.W.shape[1])
        return self.Z

    def backward(self, dZ):
        # Input dZ_prev = dl/dZ_prev
        # Output dL/dA = dL/dZ * dZ/dA  = dL/dZ * W^T
        W = self.W
        A_prev = self.layer_input
        norm=A_prev.shape[0]
        if self.trainable:
            # Gradiend update dW= dL/dW = dz/dw * dl/dz = A_prev^T dL/dz
            # Gradiend update db= dL/bb = dz/db * dl/dz = dL/dz
            dW = np.dot(A_prev.T, dZ)/norm #(2)normalize
            db = np.sum(dZ, axis=0, keepdims=True)/norm #(2)normalize
            assert dW.shape == W.shape
            assert db.shape == self.b.shape
            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db

        # Output dL/dA = dL/dZ * dZ/dA  = dL/dZ * W^T
        return np.dot(dZ, W.T) # return dA_prev



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
        act = self.activation_func(Z)
        assert Z.shape == act.shape
        return act

    def backward(self, dA):
        Z = self.layer_input
        dact = self.activation_func.gradient(Z)
        assert Z.shape == dact.shape
        assert Z.shape == dA.shape
        dZ = dact * dA
        assert(dZ.shape == (Z.shape))
        return dZ

##### Unchanged layer_input



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
        #print(dact.shape, dA.shape , Z.shape)
        dZ = np.sum(np.multiply(dact, dA[:, np.newaxis,:]), axis = 2)
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
        batch_size= Z.shape[0]
        shape = (batch_size, self.output_shape[0])
        return Z.reshape(shape)

    def backward(self, dA):
        batch_size= dA.shape[0]
        shape = (batch_size,) + self.input_shape
        return dA.reshape(shape)



class Conv2D(Layer):
    """A layer that flattens a 2D matrix
    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, input_shape= None, f = 2, n_C = 1, stride =1, padding = 0, initializer = 'normal',  lr = 0.06):
        self.layer_name= 'Conv2D'
        self.f = f
        self.initializer = initializer
        self.stride = stride
        self.pad =  padding
        self.input_shape = input_shape
        self.n_C = n_C
        self.layer_input =None
        self.trainable = True
        self.lr = lr

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



    def forward(self, A_prev, training = True):

        self.A_prev_shape = A_prev.shape #
        padding_shape = ((self.pad, self.pad), (self.pad,self.pad), (0,0), (0,0))
        A_prev_pad = np.pad(A_prev, padding_shape , 'constant', constant_values = (0,0))
        self.A_prev_pad = A_prev_pad
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
        #dZ (n_H, n_W, n_C, m)

        dW = np.zeros((self.f, self.f, self.input_shape[2], self.n_C))
        db = np.zeros((1,1,1,self.n_C))

        dA_prev_pad = np.zeros(self.A_prev_pad.shape)

        for h in range(self.output_shape[0]):        # loop over vertical axis of the output volume
            for w in range(self.output_shape[1]):    # loop over horizontal axis of the output volume
                # Find the corners of the current "slice"
                vert_start = h * self.stride
                vert_end = vert_start + self.f
                horiz_start = w * self.stride
                horiz_end = horiz_start + self.f

                A_slice = self.layer_input[vert_start:vert_end,horiz_start:horiz_end, ...] # (f, f, n_C_prev, 1, m)

                dA_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += \
                np.sum(self.W[..., np.newaxis] * ((dZ[h:h+1, w:w+1, ...])[:, :, np.newaxis, :]), axis = -2) #(f, f, n_C_prev, n_C, 1)* (1, 1, 1, n_C, m)

                dW += np.mean(A_slice * (dZ[h: h+1, w: w+1,...][:,:,np.newaxis,:, :]), axis = -1) #(f, f, n_C_prev, 1, m) * (1,1,1, n_C, m)
                db += np.mean(dZ[h, w,...], axis= -1).reshape((1,1,1,self.n_C))

        if self.trainable:
            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db

        dA_prev = dA_prev_pad[self.pad:-self.pad, self.pad:-self.pad, ...]

        assert(dA_prev.shape == self.A_prev_shape) # (n_H_prev, n_W_prev, n_C_prev, m)

        return dA_prev
