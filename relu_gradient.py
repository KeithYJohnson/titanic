import numpy as np

def relu_gradient(z):
    grad = z[z <= 0] = 0
    return z
