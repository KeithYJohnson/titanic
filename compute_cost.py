from ipdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from add_bias_column import add_bias_column
from params import *

def compute_cost(weights,
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

    a1 = add_bias_column(features)
    z2 = np.dot(a1, w2.transpose())
    a2 = sigmoid(z2)
    a2 = add_bias_column(a2)
    z3 = np.dot(a2, w3.transpose())
    a3 = sigmoid(z3)

    num_examples = y.shape[0]

    y_eq_1_term = np.multiply(-y, np.log(a3))
    y_eq_0_term = np.multiply((1 - y), np.log(1 - a3))
    unregularized_cost = (1/num_examples) * np.sum(y_eq_1_term - y_eq_0_term)

    w2_regulation_term = np.sum(w2[:,2:] ** 2);
    w3_regulation_term = np.sum(w3[:,2:] ** 2);
    regularization     = (regularization_strength / (2 * num_examples)) * (w2_regulation_term + w3_regulation_term)

    cost = unregularized_cost + regularization

    print(cost)
    return cost
