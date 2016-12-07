import pandas as pd
import time
from compute_cost import compute_cost
from rand_initialize_weights import rand_initialize_weights
from check_gradient import *
from scipy import optimize
from ipdb import set_trace as st
from params import *
from compute_gradient import *
from make_predictions import *
from impute_data import *

nontest_data = pd.DataFrame.from_csv('train.csv')
print('setting up params')
print('INPUT_LAYER_SIZE', INPUT_LAYER_SIZE)
print('NUMBER_OF_HIDDEN_UNITS', NUMBER_OF_HIDDEN_UNITS)
print('OUTPUT_LAYER', OUTPUT_LAYER)
print('REGULARIZATION_STRENGTH: ', REGULARIZATION_STRENGTH)

nontest_data = impute_data(nontest_data)

# Splitting all data into training and cross validation set
training_cv_split = round(nontest_data.shape[0] * PERCENT_TRAINING)
training_set = nontest_data[:training_cv_split]
cross_validation_set = nontest_data[training_cv_split:]

#Matching the outcomes to match training vs cross-validation features
actual_outcomes = (nontest_data.Survived.values).reshape(nontest_data.shape[0], 1)
training_set_outcomes = (training_set.Survived.values).reshape(training_set.shape[0], 1)
cross_validation_set_outcomes = (cross_validation_set.Survived.values).reshape(cross_validation_set.shape[0], 1)

theta1 = rand_initialize_weights(INPUT_LAYER_SIZE, NUMBER_OF_HIDDEN_UNITS)
theta2 = rand_initialize_weights(NUMBER_OF_HIDDEN_UNITS, OUTPUT_LAYER)
unrolled_weights = np.hstack([theta1.flatten(), theta2.flatten()])

# cost = compute_cost(unrolled_weights, training_set, actual_outcomes)
# grads = compute_gradient(unrolled_weights, training_set, actual_outcomes)
# grad_check_diff = optimize.check_grad(compute_cost, compute_gradient, unrolled_weights, training_set, actual_outcomes, epsilon = 10 ** -4)
# print('grad_check_diff: ', grad_check_diff)
# check_gradient(compute_cost, compute_gradient)

model = optimize.fmin_bfgs(compute_cost, x0=unrolled_weights, fprime=compute_gradient, args=(training_set[FEATURES_LIST].values, training_set_outcomes), full_output=1, maxiter=MAXITER)
predict(model[0], nontest_data[FEATURES_LIST].values, y=actual_outcomes)
nnparams = pd.DataFrame(model[0])
filename = "./models/params-{}-{}".format(time.strftime("%Y-%m-%d-%H%M"), MAXITER)
nnparams.to_csv(filename)

test_data = pd.read_csv('test.csv')
test_data = impute_data(test_data)
test_features = test_data[FEATURES_LIST].values
predictions = predict(model[0], test_features)


PassengerId = np.array(test_data["PassengerId"]).astype(int)
results = pd.DataFrame(predictions.astype('int'), PassengerId, columns = ["Survived"])
results_filename = filename = "./test_results/results-{}-iters{}-ils{}-hls{}-ols{}-lda{}".format(
    time.strftime("%Y-%m-%d-%H%M"),
    MAXITER,
    INPUT_LAYER_SIZE,
    NUMBER_OF_HIDDEN_UNITS,
    OUTPUT_LAYER,
    REGULARIZATION_STRENGTH
)

results.to_csv(results_filename, index_label = ["PassengerId"])
