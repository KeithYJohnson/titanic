from ipdb import set_trace as st
from approximate_gradient import *
from create_simple_nn_params import *
import numpy as np

def check_gradient(cost_function, epsilon = 10 ** -4):
    print('checking gradients')
    [features, theta1, theta2, y] = create_simple_nn_params()
    [cost, w2_grad, w3_grad] = cost_function(features, theta1, theta2, y)

    theta1_grad_appox = np.zeros(theta1.shape)
    for index, x in np.ndenumerate(theta1):
        theta1_plus = theta1
        theta1_plus[index] += epsilon
        Jplus = cost_function(features, theta1_plus, theta2, y)[0]

        theta1_minus = theta1
        theta1_minus[index] -= epsilon
        Jminus = cost_function(features, theta1_minus, theta2, y)[0]

        numerical_grad = (Jplus - Jminus) / (2 * epsilon)
        theta1_grad_appox[index] = numerical_grad

    print('checking theta1 against w2_grad, there shouldnt be a large element-wise difference')
    print(np.abs(theta1_grad_appox - w2_grad))

    theta2_grad_appox = np.zeros(theta2.shape)
    for index, x in np.ndenumerate(theta2):
        theta2_plus = theta2
        theta2_plus[index] += epsilon
        Jplus_t2 = cost_function(features, theta1, theta2_plus, y)[0]

        theta2_minus = theta2
        theta2_minus[index] -= epsilon
        Jminus_t2 = cost_function(features, theta1, theta2_minus, y)[0]

        numerical_grad_t2 = (Jplus_t2 - Jminus_t2) / (2 * epsilon)
        theta2_grad_appox[index] = numerical_grad_t2

    print('checking theta2 against w3_grad, there shouldnt be a large element-wise difference')
    print(np.abs(theta2_grad_appox - w3_grad))
