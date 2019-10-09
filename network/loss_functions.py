from __future__ import division
import numpy as np

from activation_functions import Sigmoid


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, AL):
        # Avoid division by zero
        AL = np.clip(AL, 1e-15, 1 - 1e-15)
        return - y * np.log(AL) - (1 - y) * np.log(1 - AL)

    def acc(self, y, AL):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(AL, axis=1))

    def gradient(self, y, AL):
        # Avoid division by zero
        AL = np.clip(AL, 1e-15, 1 - 1e-15)
        print  ((- (y / AL) + (1 - y) / (1 - AL) ).shape,'cross-function output dA')
        return - (y / AL) + (1 - y) / (1 - AL)
