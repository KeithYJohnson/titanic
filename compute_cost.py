from pdb import set_trace
import math
import numpy as np
from sigmoid import sigmoid

def compute_cost(features, theta1, theta2, actual_outcomes):
    # Prepend column of ones to handle the bias
    bias = np.ones((features.shape[0], 1))
    features = np.concatenate((bias,features), 1)

    first_layer_weighted_input = np.dot(features, theta1.transpose())
    first_layer_activation = sigmoid(first_layer_weighted_input)

    #Prepend column of ones to handle the bias
    second_layer_bias = np.ones((first_layer_activation.shape[0], 1))
    first_layer_activation = np.concatenate((second_layer_bias, first_layer_activation), 1)

    second_layer_weighted_input = np.dot(first_layer_activation, theta2.transpose())
    second_layer_activation = sigmoid(second_layer_weighted_input)
