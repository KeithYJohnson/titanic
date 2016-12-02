from params import *
from sigmoid import *
from ipdb import set_trace as st
from add_bias_column import *
import numpy as np

def predict(weights, features, y, input_size=INPUT_LAYER_SIZE, hidden_units=NUMBER_OF_HIDDEN_UNITS, output_size=OUTPUT_LAYER, threshold =0.5):
    w2w3_idx_boundary = ((input_size + 1) * hidden_units)
    w2 = weights[0:w2w3_idx_boundary].reshape(
        hidden_units, input_size + 1
    )

    w3 = weights[w2w3_idx_boundary:].reshape(
        output_size, hidden_units + 1
    )

    features = add_bias_column(features)
    h1 = sigmoid(np.dot(features, w2.transpose()))
    h1 = add_bias_column(h1)
    h2 = sigmoid(np.dot(h1, w3.transpose()))

    did_survive = h2 > threshold
    unique, counts = np.unique(did_survive, return_counts=True)
    print('PREDICTIONS: ', dict(zip(unique, counts)))
    print('ACCURACY: ', sum(did_survive == y) / len(y))
    return h2 > threshold
