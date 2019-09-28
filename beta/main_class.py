import numpy as np
import matplotlib.pyplot as plt
import h5py
from activation_functions import *


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
    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.loss_function = loss()
        #self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)