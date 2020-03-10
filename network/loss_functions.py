from __future__ import division
import numpy as np

from activation_functions import Sigmoid


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) /len(y_true)
    return accuracy


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return NotImplementedError()

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

    def acc(self, y, y_pred):
        return 0.5 * np.mean(np.power((y - y_pred), 2))


class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, AL):
        # Avoid division by zero
        AL = np.clip(AL, 1e-15, 1 - 1e-15)
        return - y * np.log(AL) - (1 - y) * np.log(1 - AL)

    def acc(self, y, AL):
        return accuracy_score(y[0], AL[0]>=0.5)

    def gradient(self, y, AL):
        # Avoid division by zero
        AL = np.clip(AL, 1e-15, 1 - 1e-15)
        #print  ((- (y / AL) + (1 - y) / (1 - AL) ).shape,'cross-function output dA')
        return - (y / AL) + (1 - y) / (1 - AL)



class MultiClassCrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, AL):
        # Avoid division by zero
        AL = np.clip(AL, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(AL) ,axis= 0, keepdims = True)
    #def acc(self, y, AL):
    #    return accuracy_score(y[0], AL[0]>=0.5)
    def gradient(self, y, AL):
        # Avoid division by zero
        AL = np.clip(AL, 1e-15, 1 - 1e-15)

        assert(AL.shape == (-(y / AL)).shape)
        #print  ((- (y / AL) + (1 - y) / (1 - AL) ).shape,'cross-function output dA')
        return - (y / AL)

class SoftmaxCrossEntropy(Loss):
    def __init__(self): pass

    def __call__(self, y, Z):
        # Avoid division by zero
        log_e_Z = Z- np.log(np.sum( np.exp(Z), axis=0, keepdims=True))
        return -np.sum( y * log_e_Z ,axis= 0)
    #def acc(self, y, AL):
    #    return accuracy_score(y[0], AL[0]>=0.5)
    def gradient(self, y, Z):
        p = np.exp(Z)/ np.sum( np.exp(Z), axis=0, keepdims=True)
        # Avoid division by zero
        return  -y + p
