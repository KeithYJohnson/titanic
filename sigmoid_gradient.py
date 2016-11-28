from sigmoid import sigmoid
import numpy as np

def sigmoid_gradient(z):
    sigmoided = sigmoid(z);
    return np.multiply(sigmoided, (1 - sigmoided));
