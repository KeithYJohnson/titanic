from ipdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from params import *


def compute_gradient(weights,
                    features,
                    y,
                    input_size=INPUT_LAYER_SIZE,
                    hidden_units=NUMBER_OF_HIDDEN_UNITS,
                    output_size=OUTPUT_LAYER,
                    regularization_strength=1):
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
        actual_outcome = y[row_num]
        d3 = a3 - actual_outcome # Diff bt prediction and actual outcome for the row.

        # Error of neurons in layer two
        d2 = np.multiply(
            np.dot(w3.transpose(), d3),
            sigmoid_gradient(a3)
        )

        w3_partial_deriv_wrt_j =  np.add(
            w3_partial_deriv_wrt_j,
            np.dot(d3, a2.transpose())
        )

        w2_partial_deriv_wrt_j = np.add(
            w2_partial_deriv_wrt_j,
                   #Dont care about change in bias
            np.dot(d2[1:,:], a1.transpose())
        )

    unreg_w2_grad = (1/num_examples) * w2_partial_deriv_wrt_j
    unreg_w3_grad = (1/num_examples) * w3_partial_deriv_wrt_j

    # All the 1-indexing is to ignore the bias.
    reg_w2_grad = unreg_w2_grad
    reg_w2_grad[:,1:] = reg_w2_grad[:,1:] + (regularization_strength / num_examples) * w2[:, 1:]

    reg_w3_grad = unreg_w3_grad
    reg_w3_grad[:,1:] = reg_w3_grad[:,1:] + (regularization_strength / num_examples) * w3[:, 1:]
    rolled_grads = np.hstack([reg_w2_grad.flatten(), reg_w3_grad.flatten()])

    if testing:
        return rolled_grads, reg_w3_grad, reg_w2_grad
    else:
        return rolled_grads
