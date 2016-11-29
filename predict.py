import pandas as pd
from compute_cost import compute_cost
from rand_initialize_weights import rand_initialize_weights
from check_gradient import *


training_data = pd.DataFrame.from_csv('train.csv')
# Impute missing data
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

                                                       #Parents/Children Aboard
features_list = ["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']
                                               #Siblings/Spouses Aboard
features = training_data[features_list].values

# Setup the parameters
print('setting up params')
INPUT_LAYER_SIZE  = len(features_list)
NUMBER_OF_HIDDEN_UNITS = 25
OUTPUT_LAYER = 1   # It's binary, either they survived or didn't
print('INPUT_LAYER_SIZE', INPUT_LAYER_SIZE)
print('NUMBER_OF_HIDDEN_UNITS', NUMBER_OF_HIDDEN_UNITS)
print('OUTPUT_LAYER', OUTPUT_LAYER)

theta1 = rand_initialize_weights(INPUT_LAYER_SIZE, NUMBER_OF_HIDDEN_UNITS)
print('rand_init theta1: ', theta1)
print(theta1.shape)
theta2 = rand_initialize_weights(NUMBER_OF_HIDDEN_UNITS, OUTPUT_LAYER)
print('rand_init theta2: ', theta2)
print(theta1.shape)

actual_outcomes = (training_data['Survived'].values).reshape(training_data.shape[0],1)
print('actual_outcomes.shape: ', actual_outcomes.shape)

[cost, w2_grad, w3_grad] = compute_cost(features, theta1, theta2, actual_outcomes)
print('cost: ', cost)
print('w2_grad: ', w2_grad)
print('w3_grad: ', w3_grad)
print('w2_grad.shape: ', w2_grad.shape)
print('w3_grad.shape: ', w3_grad.shape)

check_gradient(compute_cost)
