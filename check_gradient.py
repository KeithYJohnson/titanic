from ipdb import set_trace as st
from create_simple_nn_params import *
import numpy as np

def check_gradient(cost_function, grad_function, epsilon = 10 ** -4, max_diff=1e-9):
    print('checking gradients')
    [features, theta1, theta2, y] = create_simple_nn_params()
    rolled_weights = np.hstack([theta1.flatten(), theta2.flatten()])
    cost = cost_function(rolled_weights, features, y, features.shape[1], theta1.shape[0], 1)
    grads = grad_function(rolled_weights, features, y, features.shape[1], theta1.shape[0], 1)

    rolled_weights = np.zeros(rolled_weights.shape)
    for index, x in np.ndenumerate(rolled_weights):
        rolled_weights_plus = rolled_weights
        rolled_weights_plus[index] += epsilon
        Jplus = cost_function(rolled_weights_plus, features, y, features.shape[1], theta1.shape[0], 1)

        rolled_weights_minus = rolled_weights
        rolled_weights_minus[index] -= epsilon
        Jminus = cost_function(rolled_weights_minus, features, y, features.shape[1], theta1.shape[0], 1)

        numerical_grad = (Jplus - Jminus) / (2 * epsilon)
        rolled_weights[index] = numerical_grad

    print('abs element-wise diff between numerical grad against analytical grad')
    print(np.abs(rolled_weights - grads))
    print('element-wise: is the diff larger than max_diff of: ', max_diff)
    print(np.abs(rolled_weights - grads) > max_diff)
