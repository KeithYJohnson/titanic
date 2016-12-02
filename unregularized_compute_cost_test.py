from math import cos
import numpy as np
from compute_cost import *

input_layer_size  = 2
hidden_layer_size = 2
output_layer_size = 4
weights = np.array(list(range(1, 19))) / 10
features = np.cos(np.matrix('1 2 ; 3 4 ; 5 6'))
outcomes = np.matrix('4; 2; 3')
regularization_strength = 0

expected_cost =  7.4070

[actual_cost,
 actual_a1,
 actual_z2,
 actual_a2,
 actual_z3,
 actual_a3] = compute_cost(weights,
                                features,
                                outcomes,
                                input_layer_size,
                                hidden_layer_size,
                                output_layer_size,
                                regularization_strength,
                                True)

assert(np.all(abs(expected_cost - actual_cost) <= .0001))
