import numpy as np

#   Returns a matrix of size(L_out, 1 + L_in) as
#   we need an extra column to handle the bias.
def rand_initialize_weights(incoming_cxns, outgoing_cxns, handle_bias=True):
    matrix = np.zeros((outgoing_cxns, 1 + incoming_cxns))
    epsilon_init = 0.12
    matrix = np.random.rand(outgoing_cxns, handle_bias+incoming_cxns) * 2 * epsilon_init -epsilon_init
    return matrix
