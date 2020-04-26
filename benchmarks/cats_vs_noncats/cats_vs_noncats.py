

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from utils import *
import os
import sys
import matplotlib.pyplot


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
sys.path.append('../../network')


# Load data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#print('Example pic:')
# Example of a picture
#index = 11
#plt.imshow(train_x_orig[index])
#plt.show()
#print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


# Explore your dataset
train_y, test_y = train_y.T , test_y.T
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

print('After Standardize and reshape:')
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1)   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1)

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

#Define model

#Define model

n_x = 12288     # num_px * num_px * 3
n_h1 = 4
n_h2 = 7
n_h3 = 5
n_y = 1

# Model
import sys
sys.path.append('../../network')
from main_class import *
from layers import *
from loss_functions import *

md=NeuralNetwork(CrossEntropy)
np.random.seed(1)
lr = 0.0075

md.add(Dense(n_h1, input_shape=(n_x,), initializer = 'normal', lr = lr))
md.add(Activation('relu'))

md.add(Dense(n_h2, initializer = 'normal', lr = lr))
md.add(Activation('relu'))

md.add(Dense(n_h3, initializer = 'normal', lr = lr))
md.add(Activation('relu'))

md.add(Dense(n_y, initializer = 'normal', lr = lr))
md.add(Activation('sigmoid'))


#Print_network shape
md.print_network()

# Train
train, val = md.fit(train_x, train_y, n_epochs=200, batch_size=32)

#Evaluate
pred =md.predict(train_x)
pred=(pred >=0.5)
acc = np.mean((pred == train_y))
print('Training acc: {}'.format(acc))

pred_test =md.predict(test_x)>=0.5
acc = np.mean((pred_test == test_y))
print('Testing acc: {}'.format(acc))

# Print mislabelled
def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (80.0, 80.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(num_images, i + 1, 2)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
    plt.show()

#Better only do it on jupyter notebook
#print_mislabeled_images(classes, test_x, test_y, pred_test)

import imageio


my_image = "my_image4.jpg" # change this to the name of your image file
my_label_y = 0 # the true class of your image (1 -> cat, 0 -> non-cat)
fname = "images/" + my_image
image = np.array(imageio.imread(fname))
#image = np.array(imageio.imread(fname))
my_image = np.array(Image.fromarray(image).resize((num_px,num_px))).reshape((1,num_px*num_px*3))
#my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = md.predict(my_image)>=0.5
plt.imshow(image)
plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

my_image = "my_image3.jpg" # change this to the name of your image file
my_label_y = 0 # the true class of your image (1 -> cat, 0 -> non-cat)
fname = "images/" + my_image
image = np.array(imageio.imread(fname))
#image = np.array(imageio.imread(fname))
my_image = np.array(Image.fromarray(image).resize((num_px,num_px))).reshape((1,num_px*num_px*3))
#my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = md.predict(my_image)>=0.5
plt.imshow(image)
plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
