import pytest
from sigmoid_gradient import *
import numpy as np

matrix = np.matrix('-1 -2 -3; 8 1 6; 3 5 7; 4 9 2')
expected = np.matrix( \
  '1.9661e-001  1.0499e-001  4.5177e-002; \
   3.3524e-004  1.9661e-001  2.4665e-003; \
   4.5177e-002  6.6481e-003  9.1022e-004; \
   1.7663e-002  1.2338e-004  1.0499e-001'
)
actual = sigmoid_gradient(matrix)

#Basically an assertAlmostEqual
assert(np.all(abs(expected - actual) <= .0001))
