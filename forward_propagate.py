from unroll_weights import *
from add_bias_column import *
from params import *
from sigmoid import *
import numpy as np

def forward_propagate(weights,
                      features,
                      input_size=INPUT_LAYER_SIZE,
                      hidden_layer_size=NUMBER_OF_HIDDEN_UNITS,
                      output_layer_size=OUTPUT_LAYER,
                      actv_fn=ACTV_FN):

    [w2, w3] = unroll_weights(weights, input_size, hidden_layer_size, output_layer_size)

    a1 = add_bias_column(features)
    z2 = np.dot(a1, w2.transpose())
    a2 = actv_fn(z2)
    a2 = add_bias_column(a2)
    z3 = np.dot(a2, w3.transpose())
    a3 = actv_fn(z3)

    return a1, z2, a2, z3, a3
