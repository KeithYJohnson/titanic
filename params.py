                                                       #Parents/Children Aboard
FEATURES_LIST = ["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']
                                               #Siblings/Spouses Aboard
INPUT_LAYER_SIZE  = len(FEATURES_LIST)
NUMBER_OF_HIDDEN_UNITS = 25
OUTPUT_LAYER = 1   # It's binary, either they survived or didn't
MAXITER = 1000
REGULARIZATION_STRENGTH = 1
