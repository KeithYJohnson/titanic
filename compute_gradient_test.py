from math import cos
import numpy as np
from compute_gradient import *
from sigmoid import *
from sigmoid_gradient import *
from compute_cost import *

input_layer_size  = 2
hidden_layer_size = 2
output_layer_size = 4
weights  = np.array(list(range(1, 19))) / 10
features = np.cos(np.matrix('1 2 ; 3 4 ; 5 6'))
outcomes = np.matrix('4; 2; 3')
regularization_strength = 4.0

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

expected_reg_w2_grad = np.matrix('\
   2.298415  -0.082619  -0.074786; \
   2.939691  -0.107533  -0.161585  \
')

expected_reg_w3_grad = np.matrix('      \
    0.88341721  1.92598022  2.47833662; \
    0.56876234  1.94461818  2.50225374; \
    0.58466766  1.98964666  2.52643729; \
    0.59813924  2.17855173  2.72233089  \
')

expected_grad = np.matrix('\
    0.76614; \
    0.97990; \
    0.37246; \
    0.49749; \
    0.64174; \
    0.74614; \
    0.88342; \
    0.56876; \
    0.58467; \
    0.59814; \
    1.92598; \
    1.94462; \
    1.98965; \
    2.17855; \
    2.47834; \
    2.50225; \
    2.52644; \
    2.72233  \
')

expected_grads = np.array([0.76614,0.97990,0.37246,0.49749,0.64174,0.74614,0.88342,0.56876,0.58467,0.59814,1.92598,1.94462,1.98965,2.17855,2.47834,2.50225,2.52644,2.72233])

[actual_rolled_grads,
 actual_reg_w3_grad,
 actual_reg_w2_grad,
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
        actv_fn=sigmoid,
        grad_fn=sigmoid_gradient,
        testing=True
    )


def test_fn():
    assert(np.allclose(actual_rolled_grads, expected_grads))
    assert(np.allclose(expected_d3, actual_d3))
    assert(np.allclose(expected_d2, actual_d2))
    assert(np.allclose(expected_grads, actual_rolled_grads))
