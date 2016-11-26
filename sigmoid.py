import numpy as np

def sigmoid(thing):
    # TODO get working with astype
    return 1.0 / (1.0 + np.exp(-thing.astype('float')))
