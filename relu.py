import numpy as np

# A unit employing the rectifier is also called a rectified linear unit (ReLU)
def relu(z):
    activation = np.maximum(0, z).astype('float')
    return activation
