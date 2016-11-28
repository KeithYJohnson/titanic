from pdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from add_bias_column import add_bias_column

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

    y_eq_1_term = np.multiply(actual_outcomes, np.log(second_layer_activation))
    y_eq_0_term = np.multiply((1 - actual_outcomes), np.log(1 - second_layer_activation))

    num_examples = actual_outcomes.shape[0]
    cost = (1/num_examples) * np.sum(y_eq_1_term - y_eq_0_term)
