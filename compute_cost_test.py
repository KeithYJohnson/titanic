from math import cos
import numpy as np
from compute_cost import *

input_layer_size  = 2
hidden_layer_size = 2
output_layer_size = 4
weights = np.array(list(range(1, 19))) / 10
features = np.cos(np.matrix('1 2 ; 3 4 ; 5 6'))
outcomes = np.matrix('4; 2; 3')
regularization_strength = 4.0

expected_z2 = np.matrix(' \
    0.054017   0.166433; \
   -0.523820  -0.588183; \
    0.665184   0.889567'
)

expected_a2 = np.matrix(' \
    1.00000   0.51350   0.54151; \
    1.00000   0.37196   0.35705; \
    1.00000   0.66042   0.70880'
)

expected_a3 = np.matrix(' \
    0.88866   0.90743   0.92330   0.93665; \
    0.83818   0.86028   0.87980   0.89692; \
    0.92341   0.93858   0.95090   0.96085'
)

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

assert(np.all(abs(expected_z2 - actual_z2) <= .0001))
assert(np.all(abs(expected_a2 - actual_a2) <= .0001))
assert(np.all(abs(expected_a3 - actual_a3) <= .0001))
assert(np.all(abs(19.473 - actual_cost) <= .001))
