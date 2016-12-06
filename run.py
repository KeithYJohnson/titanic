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

training_data = pd.DataFrame.from_csv('train.csv')

print ('imputing missing data')
training_data["Age"] = training_data["Age"].fillna(training_data["Age"].median())
training_data["Embarked"] = training_data["Embarked"].fillna(
  training_data['Embarked'].value_counts().idxmax()
)

# Convert possible Embarked values to integers
training_data["Embarked"][training_data["Embarked"] == "S"] = 0
training_data["Embarked"][training_data["Embarked"] == "C"] = 1
training_data["Embarked"][training_data["Embarked"] == "Q"] = 2

# Convert genders to integers
training_data["Sex"][training_data["Sex"] == "female"] = 1
training_data["Sex"][training_data["Sex"] == "male"] = 0
features = training_data[FEATURES_LIST].values

# Setup the parameters
print('setting up params')
print('INPUT_LAYER_SIZE', INPUT_LAYER_SIZE)
print('NUMBER_OF_HIDDEN_UNITS', NUMBER_OF_HIDDEN_UNITS)
print('OUTPUT_LAYER', OUTPUT_LAYER)

theta1 = rand_initialize_weights(INPUT_LAYER_SIZE, NUMBER_OF_HIDDEN_UNITS)
theta2 = rand_initialize_weights(NUMBER_OF_HIDDEN_UNITS, OUTPUT_LAYER)
unrolled_weights = np.hstack([theta1.flatten(), theta2.flatten()])

actual_outcomes = (training_data['Survived'].values).reshape(training_data.shape[0], 1)

cost = compute_cost(unrolled_weights, features, actual_outcomes)
grads = compute_gradient(unrolled_weights, features, actual_outcomes)
# grad_check_diff = optimize.check_grad(compute_cost, compute_gradient, unrolled_weights, features, actual_outcomes, epsilon = 10 ** -4)
# print('grad_check_diff: ', grad_check_diff)
# check_gradient(compute_cost, compute_gradient)

model = optimize.fmin_bfgs(compute_cost, x0=unrolled_weights, fprime=compute_gradient, args=(features, actual_outcomes), full_output=1, maxiter=MAXITER)
predict(model[0], features, y=actual_outcomes)
nnparams = pd.DataFrame(model[0])
filename = "./models/params-{}-{}".format(time.strftime("%Y-%m-%d-%H%M"), MAXITER)
nnparams.to_csv(filename)


test_data = pd.read_csv('test.csv')
test_data["Age"].fillna(test_data["Age"].median())
test_data["Embarked"] = test_data["Embarked"].fillna(
  test_data['Embarked'].value_counts().idxmax()
)
test_data["Embarked"][test_data["Embarked"] == "S"] = 0
test_data["Embarked"][test_data["Embarked"] == "C"] = 1
test_data["Embarked"][test_data["Embarked"] == "Q"] = 2
test_data["Sex"][test_data["Sex"] == "female"] = 1
test_data["Sex"][test_data["Sex"] == "male"] = 0
test_features = test_data[FEATURES_LIST].values
predictions = predict(model[0], test_features)


PassengerId =np.array(test_data["PassengerId"]).astype(int)
results = pd.DataFrame(predictions.astype('int'), PassengerId, columns = ["Survived"])
results_filename = filename = "./test_results/results-{}-{}".format(time.strftime("%Y-%m-%d-%H%M"), MAXITER)
results.to_csv(results_filename, index_label = ["PassengerId"])
