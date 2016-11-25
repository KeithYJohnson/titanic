import pandas as pd
from compute_cost import compute_cost
from rand_initialize_weights import rand_initialize_weights

NUMBER_OF_HIDDEN_UNITS = 25
NUMBER_OF_CLASSIFIERS = 2

training_data = pd.DataFrame.from_csv('train.csv')
theta1 = rand_initialize_weights(training_data.shape[0], NUMBER_OF_HIDDEN_UNITS)
theta2 = rand_initialize_weights(NUMBER_OF_HIDDEN_UNITS, NUMBER_OF_CLASSIFIERS)

compute_cost(training_data, theta1, theta2)
