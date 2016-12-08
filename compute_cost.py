from ipdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from add_bias_column import add_bias_column
from params import *
from unroll_weights import *
from forward_propagate import *
from create_y_matrix import *

def compute_cost(weights,
                features,
                y,
                input_size=INPUT_LAYER_SIZE,
                hidden_units=NUMBER_OF_HIDDEN_UNITS,
                output_size=OUTPUT_LAYER,
                regularization_strength=REGULARIZATION_STRENGTH,
                actv_fn=ACTV_FN,
                grad_fn=GRAD_FN,
                testing=False):

    # 'Re-rolling' unrolled weights
    [w2, w3] = unroll_weights(weights, input_size, hidden_units, output_size)
    [a1, z2, a2, z3, a3] = forward_propagate(weights, features, input_size, hidden_units, output_size, actv_fn)
    num_examples = y.shape[0]

    y_matrix = create_y_matrix(num_examples, output_size, y)
    # st()
    y_eq_1_term = np.multiply(-y_matrix, np.log(a3))
    y_eq_0_term = np.multiply((1 - y_matrix), np.log(1 - a3))
    unregularized_cost = (1 / num_examples) * np.sum(y_eq_1_term - y_eq_0_term)

    w2_regulation_term = np.sum(w2[:,1:] ** 2);
    w3_regulation_term = np.sum(w3[:,1:] ** 2);
    regularization     = (regularization_strength / (2 * num_examples)) * (w2_regulation_term + w3_regulation_term)

    cost = unregularized_cost + regularization
    print('cost: ', cost)
    if testing:
        return cost, a1, z2, a2, z3, a3
    else:
        return cost
