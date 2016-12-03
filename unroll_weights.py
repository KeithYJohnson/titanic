
def unroll_weights(weights, input_size, hidden_layer_size, output_size):
    w2w3_idx_boundary = ((input_size + 1) * hidden_layer_size)

    w2 = weights[0:w2w3_idx_boundary].reshape(
        input_size + 1, hidden_layer_size
    ).transpose()

    w3 = weights[w2w3_idx_boundary:].reshape(
        hidden_layer_size + 1, output_size
    ).transpose()

    return [w2, w3]
