
from dnn_app_utils_v4 import *
from testCases_v4 import *



####################

layer_dims=[5,4,3]
parameters = initialize_parameters_deep(layer_dims)

for l in range(1, len(layer_dims) ):
     assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
     assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


####################

A, W, b = linear_forward_test_case()

Z, linear_cache = linear_forward(A, W, b)
assert(Z.shape == (W.shape[0], A.shape[1]))


####################

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")

(A.shape == (W.shape[0], A_prev.shape[1]))


####################

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)

assert(AL.shape == (1,X.shape[1]))


####################

dZ, linear_cache = linear_backward_test_case()
A_prev, W, b = linear_cache 
dA_prev, dW, db = linear_backward(dZ, linear_cache)


assert (dA_prev.shape == A_prev.shape)
assert (dW.shape == W.shape)
assert (db.shape == b.shape)

####################


dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
assert (dA_prev.shape == A_prev.shape)
assert (dW.shape == W.shape)
assert (db.shape == b.shape)

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
assert (dA_prev.shape == A_prev.shape)
assert (dW.shape == W.shape)
assert (db.shape == b.shape)

####################

AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)



parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

for i in parameters.keys():
    assert (parameters[i].shape == grads['d'+i].shape)

print('\n ok \n')

