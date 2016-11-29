import numpy as np
from rand_initialize_weights import *

def create_simple_nn_params(num_examples=4, input_layer_size=3, hidden_layer_size=5):
    # must always be one cuz compute_cost doesnt 1 vs many
    OUTPUT_LAYER_SIZE = 1

    theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    theta2 = rand_initialize_weights(hidden_layer_size, OUTPUT_LAYER_SIZE)

    # Reusing rand_initialize_weights for generating features; should be fine.
    features = rand_initialize_weights(input_layer_size, num_examples, handle_bias=False)

    y = np.random.randint(2, high=None, size=num_examples, dtype='l').reshape(num_examples, 1)
    return features, theta1, theta2, y
