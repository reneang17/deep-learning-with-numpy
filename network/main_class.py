import numpy as np
import matplotlib.pyplot as plt
import h5py
from batcher import *

import time
from tqdm import tqdm



class NeuralNetwork():
    """Neural Network.
    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels (X, y)
    """
    def __init__(self, Loss_function, validation_data=None):
        self.layers = []
        self.loss_function = Loss_function()
        self.errors = {"training": [], "validation": []}

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y": y}

    def add(self, layer):
        """ Method which adds a layer to the neural network """
        # If not first layer added then set the input shape
        # to the output shape of the last added layer

        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].get_output_shape())

        # If the layer has weights that needs to be initialized
        if hasattr(layer, 'initialize'):
            layer.initialize()

        # Add layer to the network
        self.layers.append(layer)

    def train_on_batch(self, X, y):
        """ Forward/backward on batch """
        y_pred = self._forward(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))#(1)mean normalizes
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate. Update weights
        self._backward(loss_grad=loss_grad)

        return loss, acc

    def test_on_batch(self, X, y):
        """ Test on batch """
        y_pred = self._forward(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size):
        """ Train on n_epochs """

        for _ in tqdm(range(n_epochs)):
            time.sleep(0)

            batch_error = []

            batch_iterator =  Batcher(X, y, batch_size=batch_size)

            for X_batch, y_batch in batch_iterator:

                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)

            self.errors["training"].append(np.mean(batch_error))

            if self.val_set is not None:
                val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                self.errors["validation"].append(val_loss)

        return self.errors["training"], self.errors["validation"]

    def predict(self, X):
        """ Use the trained model to predict labels of X """
        return self._forward(X, training=False)



    def _forward(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(layer_output, training)


        return layer_output

    def _backward(self, loss_grad):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        #print(loss_grad.shape)
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
            #print(loss_grad.shape)

    def print_network(self):
        print("***** Current network *****")
        print('layer', '\t\t', 'output_shape', '\t\t', 'Input_shape')
        for layer in self.layers:
            print(layer.layer_name, '\t\t', layer.output_shape, '\t\t',layer.input_shape)
        print('Loss funciton ', '\t\t', self.loss_function.loss_name)
        print("***************************")
