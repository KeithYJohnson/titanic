from params import *
from sigmoid import *
from ipdb import set_trace as st
from add_bias_column import *
from unroll_weights import *
import numpy as np

def predict(weights, features, input_size=INPUT_LAYER_SIZE, hidden_units=NUMBER_OF_HIDDEN_UNITS, output_size=OUTPUT_LAYER, threshold =0.5, y=None):
    [w2, w3] = unroll_weights(weights, input_size, hidden_units, output_size)

    features = add_bias_column(features)
    h1 = sigmoid(np.dot(features, w2.transpose()))
    h1 = add_bias_column(h1)
    h2 = sigmoid(np.dot(h1, w3.transpose()))

    did_survive = h2 > threshold
    unique, counts = np.unique(did_survive, return_counts=True)

    if not y == None:
        print('ACCURACY: ', sum(did_survive == y) / len(y))

    print('PREDICTIONS: ', dict(zip(unique, counts)))

    return h2 > threshold
