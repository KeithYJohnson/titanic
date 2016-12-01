from ipdb import set_trace as st
from create_simple_nn_params import *
import numpy as np

def check_gradient(cost_function, grad_function, epsilon = 10 ** -4, max_diff=1e-9):
    print('checking gradients')
    [features, theta1, theta2, y] = create_simple_nn_params()
    unrolled_weights = np.hstack([theta1.flatten(), theta2.flatten()])
    cost = cost_function(unrolled_weights, features, y, features.shape[1], theta1.shape[0], 1)
    grads = grad_function(unrolled_weights, features, y, features.shape[1], theta1.shape[0], 1)

    unrolled_weights_approx = np.zeros(unrolled_weights.shape)
    for index, x in np.ndenumerate(unrolled_weights):
        unrolled_weights_plus = unrolled_weights
        unrolled_weights_plus[index] += epsilon
        Jplus = cost_function(unrolled_weights_plus, features, y, features.shape[1], theta1.shape[0], 1)

        unrolled_weights_minus = unrolled_weights
        unrolled_weights_minus[index] -= epsilon
        Jminus = cost_function(unrolled_weights_minus, features, y, features.shape[1], theta1.shape[0], 1)

        numerical_grad = (Jplus - Jminus) / (2 * epsilon)
        unrolled_weights_approx[index] = numerical_grad

    print('abs element-wise diff between numerical grad against analytical grad')
    print(np.abs(unrolled_weights_approx - grads))
    print('element-wise: is the diff larger than max_diff of: ', max_diff)
    print(np.abs(unrolled_weights_approx - grads) > max_diff)
