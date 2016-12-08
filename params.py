from sigmoid import *
from sigmoid_gradient import *
from relu import *
from relu_gradient import *

                                                       #Parents/Children Aboard
FEATURES_LIST = ["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']
                                               #Siblings/Spouses Aboard
INPUT_LAYER_SIZE  = len(FEATURES_LIST)
NUMBER_OF_HIDDEN_UNITS = 25
OUTPUT_LAYER = 1   # It's binary, either they survived or didn't
MAXITER = 200
REGULARIZATION_STRENGTH = 0

# split between training and cross validation data
PERCENT_TRAINING = .9
PERCENT_CV = 1 - PERCENT_TRAINING

# Activation function
ACTV_FN = sigmoid
GRAD_FN = eval("{}_gradient".format(ACTV_FN.__name__))
