import numpy as np
from unroll_weights import *

input_layer_size = 2
hidden_layer_size = 2
output_size = 4
weights = np.array(list(range(1, 19))) / 10

expected_w2 = np.matrix( \
  '0.10000   0.30000   0.50000; \
   0.20000   0.40000   0.60000'
)

expected_w3 = np.matrix( \
  '0.70000   1.10000   1.50000; \
   0.80000   1.20000   1.60000; \
   0.90000   1.30000   1.70000; \
   1.00000   1.40000   1.80000'
)

[actual_w2, actual_w3] = unroll_weights(
    weights,
    input_layer_size,
    hidden_layer_size,
    output_size
)

assert(np.all(np.equal(actual_w2, expected_w2)))
assert(np.all(np.equal(actual_w3, expected_w3)))
