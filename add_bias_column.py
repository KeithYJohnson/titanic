from numpy import ones, concatenate

def add_bias_column(thing):
    bias = ones((thing.shape[0], 1))
    return concatenate((bias, thing), 1)
