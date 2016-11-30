from ipdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from add_bias_column import add_bias_column
from params import *

def compute_cost(weights, features, y):
    a1 = add_bias_column(features)
    z2 = np.dot(a1, w2.transpose())
    a2 = sigmoid(z2)
    a2 = add_bias_column(a2)
    z3 = np.dot(a2, w3.transpose())
    a3 = sigmoid(z3)
    z3 = np.dot(a2, w3.transpose())
    a3 = sigmoid(z3) # Predictions

    y_eq_1_term = np.multiply(y, np.log(a3))
    y_eq_0_term = np.multiply((1 - y), np.log(1 - a3))
    num_examples = y.shape[0]
    # Vectorized Cost
    cost = (1/num_examples) * np.sum(y_eq_1_term - y_eq_0_term)

    #Accumulators used to calculate the partial derivative of J(Theta)
    w3_partial_deriv_wrt_j = np.zeros(w3.shape)
    w2_partial_deriv_wrt_j = np.zeros(w2.shape)

    #Iterative backprop, TODO vectorize
    for row_num in range(0, a1.shape[0]):
        #Get row, prepend a 1 as the bias, and then reshape into 8d vector
        num_rows = features.shape[1]
        a1 = np.insert(features[row_num, :], 0, 1, 0).reshape(num_rows + 1,1)
        z2 = np.dot(w2, a1)
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1, 0)
        z3 = np.dot(w3, a2)
        a3 = sigmoid(z3)

        #Error of the neurons in layer 3
        d3 = a3 - y[row_num, :] # Diff bt prediction and actual outcome for the row.

        a3_deriv = sigmoid_gradient(z3)
        # Error of neurons in layer two
        d2 = np.multiply(
            np.dot(w3.transpose(), d3),
            a3_deriv
        )

        w3_partial_deriv_wrt_j =  np.add(
            w3_partial_deriv_wrt_j,
            np.dot(d3, a2.transpose())
        )

        w2_partial_deriv_wrt_j = np.add(
            w2_partial_deriv_wrt_j,
                   #Dont care about change in bias
            np.dot(d2[1:d2.shape[0],:], a1.transpose())
        )

    w2_grad = (1/num_examples) * w2_partial_deriv_wrt_j
    w3_grad = (1/num_examples) * w3_partial_deriv_wrt_j

    return cost, w2_grad, w3_grad
