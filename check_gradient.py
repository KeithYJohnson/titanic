from ipdb import set_trace as st
from create_simple_nn_params import *
import numpy as np

def check_gradient(cost_function, grad_function, epsilon = 10 ** -4, max_diff=1e-9):
    print('checking gradients')
    [features, theta1, theta2, y, input_layer_size, hidden_layer_size, output_layer_size] = create_simple_nn_params()
    rolled_weights = np.hstack([theta1.T.flatten(), theta2.T.flatten()])
    cost = cost_function(rolled_weights, features, y, input_layer_size, hidden_layer_size, output_layer_size)
    grads = grad_function(rolled_weights, features, y, input_layer_size, hidden_layer_size, output_layer_size)

    rolled_weights_appox = np.zeros(rolled_weights.shape)
    perturb = np.zeros(rolled_weights.shape)
    for index, x in np.ndenumerate(rolled_weights_appox):
        perturb[index] = epsilon

        Jplus = cost_function((rolled_weights + perturb), features, y, input_layer_size, hidden_layer_size, output_layer_size)
        Jminus = cost_function((rolled_weights - perturb), features, y, input_layer_size, hidden_layer_size, output_layer_size)

        numerical_grad = (Jplus - Jminus) / (2 * epsilon)
        rolled_weights_appox[index] = numerical_grad
        perturb[index] = 0

    print('abs element-wise diff between numerical grad against analytical grad')
    print(np.abs(rolled_weights_appox - grads))
    print('\nnp.linalg.norm(rolled_weights - grads)/np.linalg.norm(rolled_weights + grads);')
    print('should be less than the max_diff of: ', max_diff)
    diff = np.linalg.norm(rolled_weights_appox - grads)/np.linalg.norm(rolled_weights_appox + grads);
    print('diff: ', diff, '\n\n')
