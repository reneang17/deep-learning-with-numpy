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

def load_mnist():
    if not os.path.exists(os.path.join(os.curdir, "data_mnist")):
        os.mkdir(os.path.join(os.curdir, "data_mnist"))
        wget.download("http://deeplearning.net/data/mnist/mnist.pkl.gz", out="data_mnist")

    data_file = gzip.open(os.path.join(os.curdir, "data_mnist", "mnist.pkl.gz"), "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = np.array([np.reshape(x, (784, 1)) for x in train_data[0]])
    train_results = np.array([vectorized_result(y) for y in train_data[1]])
    train_data  = train_inputs.squeeze().T, \
    train_results.squeeze().T

    val_inputs = np.array([np.reshape(x, (784, 1)) for x in val_data[0]])
    val_results = np.array([vectorized_result(y) for y in val_data[1]])
    val_data  = val_inputs.squeeze().T, \
    val_results.squeeze().T

    test_inputs = np.array([np.reshape(x, (784, 1)) for x in test_data[0]])
    test_results = np.array([vectorized_result(y) for y in test_data[1]])
    test_data  = test_inputs.squeeze().T, \
    test_results.squeeze().T

    return train_data, val_data, test_data


train_data, val_data, test_data = load_mnist()

train_x , train_y = train_data
val_x , val_y = val_data
test_x , test_y = test_data


# Building network

from main_class import *
from layers import *
from loss_functions import *

print('MultiClassCrossEntropy')
md=NeuralNetwork(MultiClassCrossEntropy)
np.random.seed(1)

n_x = 784    # num_px * num_px * 3
lr = 0.05    # num_px * num_px * 3
md.add(Dense(100, input_shape=(n_x,), initializer = 'normal', lr = lr))
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
md.add(Dense(100, input_shape=(n_x,), initializer = 'normal', lr = lr))
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
