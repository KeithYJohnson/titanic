from pdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from params import *
from unroll_weights import *
from add_bias_column import *


def compute_gradient(weights,
                    features,
                    y,
                    input_size=INPUT_LAYER_SIZE,
                    hidden_units=NUMBER_OF_HIDDEN_UNITS,
                    output_size=OUTPUT_LAYER,
                    regularization_strength=1,
                    testing=False):

    [w2, w3] = unroll_weights(weights, input_size, hidden_units, output_size)
    num_examples = features.shape[0]

    #Accumulators used to calculate the partial derivative of J(Theta)
    w3_partial_deriv_wrt_j = np.zeros(w3.shape)
    w2_partial_deriv_wrt_j = np.zeros(w2.shape)

    #Iterative backprop, TODO vectorize
    num_features = features.shape[1]


    #Testing vectorized backprop here throwaway if its not in working shape
    # a1 = add_bias_column(features)
    # z2 = np.dot(a1, w2.transpose())
    # a2 = sigmoid(z2)
    # a2 = add_bias_column(a2)
    # z3 = np.dot(a2, w3.transpose())
    # a3 = sigmoid(z3)
    # d3 = np.multiply(a3 - y, sigmoid_gradient(z3))
    # d2 = np.multiply(
    #     np.dot(w3.T, d3.T),
    #     add_bias_column(sigmoid_gradient(z2))
    #     # sigmoid_gradient(z2)
    # )
    # st()
    # d2 = np.multily()

    for row_num in range(0, features.shape[0]):
        #Get row, prepend a 1 as the bias, and then reshape into 8d vector
        row = np.array(features[row_num, :].reshape(num_features, 1))
        a1 = np.insert(row, 0, 1, 0)
        z2 = np.dot(w2, a1)
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1, 0)
        z3 = np.dot(w3, a2)
        a3 = sigmoid(z3)
        #Error of the neurons in layer 3
        actual_outcome = y[row_num]
        classes = np.array(range(1, output_size + 1))
        y_matrix = classes == actual_outcome;

        d3 = a3 - y_matrix.T # Diff bt predictions and actual outcome for the row.

        # Error of neurons in layer two
        d2 = np.multiply(
            np.dot(w3.T, d3),
            np.insert(sigmoid_gradient(z2),0,1,0)
        )
        w3_partial_deriv_wrt_j =  np.add(
            w3_partial_deriv_wrt_j,
            np.dot(d3, a2.T)
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
