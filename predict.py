import pandas as pd
from compute_cost import compute_cost
from rand_initialize_weights import rand_initialize_weights


training_data = pd.DataFrame.from_csv('train.csv')
# Impute missing data
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
INPUT_LAYER_SIZE  = len(features_list)
NUMBER_OF_HIDDEN_UNITS = 25
NUMBER_OF_CLASSIFIERS = 2   # It's binary, either they survived or didn't

theta1 = rand_initialize_weights(INPUT_LAYER_SIZE, NUMBER_OF_HIDDEN_UNITS)
theta2 = rand_initialize_weights(NUMBER_OF_HIDDEN_UNITS, NUMBER_OF_CLASSIFIERS)

actual_outcomes = (training_data['Survived'].values).reshape(training_data.shape[0],1)
compute_cost(features, theta1, theta2, actual_outcomes)
