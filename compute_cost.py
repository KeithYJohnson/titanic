from pdb import set_trace as st
import math
import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient
from add_bias_column import add_bias_column

def compute_cost(features, w2, w3, y):
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


    #Iterative backprop, TODO vectorize
    for row_num in range(0, a1.shape[0]):
        #Get row, prepend a 1 as the bias, and then reshape into 8d vector
        a1 = np.insert(features[row_num, :], 0, 1, 0).reshape(8,1)
        z2 = np.dot(w2, a1)
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1, 0)
        z3 = np.dot(w3, a2)
        a3 = sigmoid(z3)


