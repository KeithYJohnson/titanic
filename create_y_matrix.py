import numpy as np

def create_y_matrix(num_examples, output_size, y):
    y_matrix = np.zeros((num_examples, output_size))
    classes = np.array(range(1, output_size + 1))
    for i in range(num_examples):
        outcome = y[i]
        y_matrix[i, outcome - 1] = 1

    return y_matrix
