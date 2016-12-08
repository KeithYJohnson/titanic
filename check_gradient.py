from ipdb import set_trace as st
from create_simple_nn_params import *
import numpy as np

def check_gradient(cost_function, grad_function, epsilon = 10 ** -4, max_diff=1e-9):
    print('checking gradients')
    [features, theta1, theta2, y, input_layer_size, hidden_layer_size, output_layer_size] = create_simple_nn_params()
    rolled_weights = np.hstack([theta1.T.flatten(), theta2.T.flatten()])
    cost = cost_function(rolled_weights, features, y, input_layer_size, hidden_layer_size, output_layer_size, 4)
    analytical_grad = grad_function(rolled_weights, features, y, input_layer_size, hidden_layer_size, output_layer_size, 4)

    numerical_grad = np.zeros(rolled_weights.shape)
    perturb = np.zeros(rolled_weights.shape)
    for index, x in np.ndenumerate(numerical_grad):
        perturb[index] = epsilon

        Jplus = cost_function((rolled_weights + perturb), features, y, input_layer_size, hidden_layer_size, output_layer_size)
        Jminus = cost_function((rolled_weights - perturb), features, y, input_layer_size, hidden_layer_size, output_layer_size)

        weight_grad = (Jplus - Jminus) / (2 * epsilon)
        numerical_grad[index] = weight_grad
        perturb[index] = 0

    print('abs element-wise diff between numerical grad against analytical grad')
    print(np.abs(numerical_grad - analytical_grad))
    relative_error_diff(numerical_grad, analytical_grad)
    norm_diff(numerical_grad, analytical_grad, max_diff)





## SEE http://cs231n.github.io/neural-networks-3/
def relative_error_diff(numerical_grad, analytical_grad):
    diff = abs(analytical_grad - numerical_grad) / np.maximum(abs(numerical_grad), abs(analytical_grad))
    print("relative_error_diff: ", diff)

def norm_diff(numerical_grad, analytical_grad, max_diff):
    print('\nnp.linalg.norm(rolled_weights - analytical_grad)/np.linalg.norm(rolled_weights + analytical_grad);')
    print('should be less than the max_diff of: ', max_diff)
    diff = np.linalg.norm(numerical_grad - analytical_grad) / np.linalg.norm(numerical_grad + analytical_grad);
    print('diff: ', diff, '\n\n')
