from compute_cost import *
from compute_gradient import *
from check_gradient import *
from create_simple_nn_params import *
from scipy import optimize
import numpy as np

print('running handolled gradient checking')
check_gradient(compute_cost, compute_gradient)

[features, theta1, theta2, y, input_layer_size, hidden_layer_size, output_layer_size] = create_simple_nn_params()
print('running scipy.optimize.check_gradient')

grad_check_diff = optimize.check_grad(
    compute_cost,
    compute_gradient,
    np.hstack([theta1.flatten(), theta2.flatten()]),
    features,
    y,
    input_layer_size,
    hidden_layer_size,
    output_layer_size,
    epsilon = 10 ** -4
)

print('grad_check_diff: ', grad_check_diff)
