from ipdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from params import *


def compute_gradient(weights, features, y, input_size=INPUT_LAYER_SIZE, hidden_units=NUMBER_OF_HIDDEN_UNITS, output_size=OUTPUT_LAYER):
    # 'Re-rolling' unrolled weights
    w2w3_idx_boundary = ((input_size + 1) * hidden_units)
    w2 = weights[0:w2w3_idx_boundary].reshape(
        hidden_units, input_size + 1
    )

    w3 = weights[w2w3_idx_boundary:].reshape(
        output_size, hidden_units + 1
    )

    num_examples = features.shape[0]

    #Accumulators used to calculate the partial derivative of J(Theta)
    w3_partial_deriv_wrt_j = np.zeros(w3.shape)
    w2_partial_deriv_wrt_j = np.zeros(w2.shape)

    #Iterative backprop, TODO vectorize
    for row_num in range(0, features.shape[0]):
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

    return np.hstack([w2_grad.flatten(), w3_grad.flatten()])
