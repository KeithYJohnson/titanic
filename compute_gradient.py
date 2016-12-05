from pdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from params import *
from unroll_weights import *
from add_bias_column import *
from forward_propagate import *
from create_y_matrix import *

def compute_gradient(weights,
                    features,
                    y,
                    input_size=INPUT_LAYER_SIZE,
                    hidden_units=NUMBER_OF_HIDDEN_UNITS,
                    output_size=OUTPUT_LAYER,
                    regularization_strength=1,
                    testing=False):

    [w2, w3] = unroll_weights(weights, input_size, hidden_units, output_size)
    [a1, z2, a2, z3, a3] = forward_propagate(weights, features, input_size, hidden_units, output_size)

    num_examples = features.shape[0]
    num_features = features.shape[1]

    y_matrix = create_y_matrix(num_examples, output_size, y)
    d3 = a3 - y_matrix

    # δ**l=((w**l+1).T * δ**l+1)⊙ σ′(z**l),
    d2 = np.multiply(
        np.dot(d3, w3[:,1:]),
        sigmoid_gradient(z2)
    )

    unreg_w2_grad = (1/num_examples) * np.dot(d2.T, a1)
    unreg_w3_grad = (1/num_examples) * np.dot(d3.T, a2)

    w2[:,0] = 0
    w3[:,0] = 0
    w2_penalty = w2 * (regularization_strength / num_examples)
    w3_penalty = w3 * (regularization_strength / num_examples)
    reg_w2_grad = unreg_w2_grad + w2_penalty
    reg_w3_grad = unreg_w3_grad + w3_penalty
    rolled_grads = np.hstack([reg_w2_grad.T.flatten(), reg_w3_grad.T.flatten()])

    if testing:
        return rolled_grads, reg_w3_grad, reg_w2_grad, unreg_w3_grad, unreg_w2_grad, d3, d2
    else:
        return rolled_grads
