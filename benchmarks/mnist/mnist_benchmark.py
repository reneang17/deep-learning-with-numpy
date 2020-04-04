import gzip
import os
import pickle
import sys
import wget
import numpy as np
sys.path.append('../../network')

# Loading data

def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

from tensorflow import keras

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    y_train = np.array([vectorized_result(y) for y in y_train])
    y_test = np.array([vectorized_result(y) for y in y_test])

    y_train = np.transpose(y_train.squeeze(),(1,0))
    y_test = np.transpose(y_test.squeeze(),(1,0))


    #reshape
    X_train = np.transpose(X_train, (1, 2, 0))
    X_test = np.transpose(X_test, (1, 2, 0))

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:, :, :-10000], X_train[:, :, -10000:]
    y_train, y_val = y_train[:, :-10000], y_train[:, -10000:]

    if flatten:
        X_train = X_train.reshape((28*28, X_train.shape[-1]))
        X_val = X_val.reshape((28*28, X_val.shape[-1]))
        X_test = X_test.reshape((28*28, X_test.shape[-1]))

    return X_train, y_train, X_val, y_val, X_test, y_test

train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(True)

# Building network
from main_class import *
from layers import *
from loss_functions import *

print('MultiClassCrossEntropy')
md=NeuralNetwork(MultiClassCrossEntropy)
np.random.seed(1)

n_x = 784    # num_px * num_px * 3
lr = 0.05    # num_px * num_px * 3
md.add(Flatten(input_shape = (28, 28, )))
md.add(Dense(100, initializer = 'normal', lr = lr))
md.add(Activation('relu'))
md.add(Dense(200, initializer = 'normal', lr = lr))
md.add(Activation('relu'))
md.add(Dense(10, initializer = 'normal', lr = lr))
md.add(Activation_SoftMax())

md.print_network()

#train
hist = md.fit(train_x, train_y, n_epochs=25, batch_size=32)

def softmax(x):
        e_x = np.exp(x )
        return e_x / np.sum(e_x, axis=0, keepdims=True)

def accuracy(test_x, test_y):
    preds = md.predict(test_x)
    preds = np.array([y for y in np.argmax(preds, axis=0)]).squeeze()
    test_y_ = np.array([y for y in np.argmax(test_y, axis=0)]).squeeze()
    return np.mean(preds == test_y_)

## Evalaution
print('Training accuarecy: {}'.format(accuracy(train_x , train_y)))
print('Test accuarecy: {}'.format(accuracy(test_x , test_y)))
print('Training loss: {}'.format(hist[0][-1]))


print('Using')
md=NeuralNetwork(SoftmaxCrossEntropy)
np.random.seed(1)
lr = 0.05
n_x = 784    # num_px * num_px * 3
md.add(Flatten(input_shape = (28, 28,)))
md.add(Dense(100, initializer = 'normal', lr = lr))
md.add(Activation('relu'))
md.add(Dense(200, initializer = 'normal', lr = lr))
md.add(Activation('relu'))
md.add(Dense(10, initializer = 'normal', lr = lr))

md.print_network()


#train
hist = md.fit(train_x, train_y, n_epochs=25, batch_size=32)

def softmax(x):
        e_x = np.exp(x )
        return e_x / np.sum(e_x, axis=0, keepdims=True)

def accuracy(test_x, test_y):
    preds = softmax(md.predict(test_x))
    preds = np.array([y for y in np.argmax(preds, axis=0)]).squeeze()
    test_y_ = np.array([y for y in np.argmax(test_y, axis=0)]).squeeze()
    return np.mean(preds == test_y_)

## Evalaution
print('Training accuarecy: {}'.format(accuracy(train_x , train_y)))
print('Test accuarecy: {}'.format(accuracy(test_x , test_y)))
print('Training loss: {}'.format(hist[0][-1]))
