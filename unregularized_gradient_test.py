from math import cos
import numpy as np
from compute_gradient import *

input_layer_size  = 2
hidden_layer_size = 2
output_layer_size = 4
weights  = np.array(list(range(1, 19))) / 10
features = np.cos(np.matrix('1 2 ; 3 4 ; 5 6'))
outcomes = np.matrix('4; 2; 3')
regularization_strength = 0

expected_grad =np.array([0.766138,0.979897-0.027540-0.035844-0.024929-0.053862,0.883417,0.568762,0.584668,0.598139,0.459314,0.344618,0.256313,0.311885,0.478337,0.368920,0.259771,0.322331])

expected_d2 = np.matrix('\
   0.79393   1.05281; \
   0.73674   0.95128; \
   0.76775   0.93560  \
')

expected_d3 = np.matrix('\
   0.888659   0.9074k27   0.923305  -0.063351; \
   0.838178  -0.139718   0.879800   0.896918; \
   0.923414   0.938578  -0.049102   0.960851  \
')

expected_unreg_w2_grad = np.matrix('\
   2.298415  -0.082619  -0.074786; \
   2.939691  -0.107533  -0.161585  \
')
expected_unreg_w3_grad = np.matrix('\
   2.65025   1.37794   1.43501; \
   1.70629   1.03385   1.10676; \
   1.75400   0.76894   0.77931; \
   1.79442   0.93566   0.96699  \
')
expected_grads = np.array([0.766138, 0.979897, -0.027540, -0.035844, -0.024929, -0.053862, 0.883417, 0.568762, 0.584668, 0.598139, 0.459314, 0.344618, 0.256313, 0.311885, 0.478337, 0.368920, 0.259771, 0.322331]).reshape(1,18)
[actual_rolled_grads,
 actual_w3_grad,
 actual_w2_grad,
 actual_unreg_w3_grad,
 actual_unreg_w2_grad,
 actual_d3,
 actual_d2
] = compute_gradient(
        weights,
        features,
        outcomes,
        input_layer_size,
        hidden_layer_size,
        output_layer_size,
        regularization_strength,
        True
    )


assert(np.all(abs(expected_d3 - actual_d3) <= .0001))
assert(np.all(abs(expected_d2 - actual_d2) <= .0001))
assert(np.all(abs(expected_unreg_w3_grad - actual_unreg_w3_grad) <= .0001))
assert(np.all(abs(expected_unreg_w2_grad - actual_unreg_w2_grad) <= .0001))
assert(np.all(abs(actual_rolled_grads - expected_grads) <= .0001))
#
