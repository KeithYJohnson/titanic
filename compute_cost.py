from ipdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from add_bias_column import add_bias_column
from params import *

def compute_cost(weights, features, y):
    # 'Re-rolling' unrolled weights
    w2w3_idx_boundary = ((INPUT_LAYER_SIZE + 1) * NUMBER_OF_HIDDEN_UNITS)
    w2 = weights[0:w2w3_idx_boundary].reshape(
        NUMBER_OF_HIDDEN_UNITS, INPUT_LAYER_SIZE + 1
    )

    w3 = weights[w2w3_idx_boundary:].reshape(
        OUTPUT_LAYER, NUMBER_OF_HIDDEN_UNITS + 1
    )

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
    print(cost)
    return cost
