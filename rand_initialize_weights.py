import numpy as np

#   Returns a matrix of size(L_out, 1 + L_in) as
#   we need an extra column to handle the bias.
def rand_initialize_weights(incoming_cxns, outgoing_cxns, handle_bias=True):
    if handle_bias:
        incoming_cxns += 1

    return np.random.rand(outgoing_cxns, incoming_cxns)
