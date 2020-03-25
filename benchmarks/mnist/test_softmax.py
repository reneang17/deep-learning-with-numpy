import numpy as np
import sys
sys.path.append('../../network')
from activation_functions import Softmax

soft = Softmax()

# Testing derivative



test_matrix = np.random.rand(5,3)
test_matrix.shape

Delta= 0.000000001

displaced = np.zeros(test_matrix.shape)
displaced[:,:] = test_matrix
displaced[np.arange(0,1), :] =displaced[np.arange(0,1), :]  + Delta

ans = ((soft(displaced) -  soft(test_matrix)  )/Delta  ) [:,:]- \
( soft.gradient(test_matrix)   )[:,0,:] < 0.0000001
print(ans)


displaced = np.zeros(test_matrix.shape)
displaced[:,:] = test_matrix
displaced[np.arange(2,3), :] =displaced[np.arange(2,3), :]  + Delta
ans = ((soft(displaced) -  soft(test_matrix)  )/Delta  ) [:,:]- \
( soft.gradient(test_matrix)   )[:,2,:] < 0.0000001
print(ans)

# Cross checking soft function with and without soft max included

from loss_functions import MultiClassCrossEntropy
import numpy as np
nb_classes = 5
n_batch = 3
targets = np.array([[0, n_batch,0]]).reshape(-1)
one_hot_targets = np.eye(nb_classes)[targets].T
test_AL = soft(np.random.random((nb_classes,n_batch)))


# Direct method
multi  = MultiClassCrossEntropy()
direct = multi.loss(one_hot_targets , soft(test_AL))
print(direct[0])

#soft max then loss function
from loss_functions import SoftmaxCrossEntropy
softmulti = SoftmaxCrossEntropy()
indirect = softmulti.loss(one_hot_targets , test_AL)
print(indirect)

# Test back prop of softmax layer
direct = softmulti.gradient(one_hot_targets , test_AL)
print(direct)

from layers import Activation_SoftMax
actsoft = Activation_SoftMax()
actsoft.forward(test_AL)
indirect = actsoft.backward(multi.gradient(one_hot_targets , soft(test_AL) ))
print(indirect)

# Calculating first derivative numerically

displaced = np.zeros(test_AL.shape)
displaced[:,:] = test_AL
displaced[np.arange(0,1), :] = displaced[np.arange(0,1), :]  + Delta
numerically = (softmulti.loss(one_hot_targets , displaced) - softmulti.loss(one_hot_targets , test_AL))/Delta
print(numerically)
