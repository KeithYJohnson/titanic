from ipdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from params import *


def compute_gradient(weights, features, y):
    # w2 = weights[0:((INPUT_LAYER_SIZE + 1) * NUMBER_OF_HIDDEN_UNITS)].reshape(
    #     NUMBER_OF_HIDDEN_UNITS, INPUT_LAYER_SIZE + 1
    # )
    # w3 = weights[0:((NUMBER_OF_HIDDEN_UNITS + 1) * OUTPUT_LAYER)].reshape(
    #     OUTPUT_LAYER, NUMBER_OF_HIDDEN_UNITS + 1
    # )

    print('shape c_grad weights: ', weights.shape)
    print('c_grad weights.size: ', weights.size)
    w2 = weights[:200].reshape(25,8)
    w3 = weights[200:].reshape(1,26)
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
