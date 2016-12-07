from params import *
from sigmoid import *
from ipdb import set_trace as st
from add_bias_column import *
from unroll_weights import *
import numpy as np
import pandas as pd
import time

def predict(weights, features, input_size=INPUT_LAYER_SIZE, hidden_units=NUMBER_OF_HIDDEN_UNITS, output_size=OUTPUT_LAYER, threshold =0.5, y=None):
    [w2, w3] = unroll_weights(weights, input_size, hidden_units, output_size)

    features = add_bias_column(features)
    h1 = sigmoid(np.dot(features, w2.transpose()))
    h1 = add_bias_column(h1)
    h2 = sigmoid(np.dot(h1, w3.transpose()))

    did_survive = h2 > threshold
    unique, counts = np.unique(did_survive, return_counts=True)
    df = pd.DataFrame(y, columns= ["outcome"])
    df['predictions'] = did_survive.flatten()
    df['correct'] = df['predictions'] == df['outcome']

    if not y == None:
        print('ACCURACY: ', sum(did_survive == y) / len(y))
        true_positives  = np.where(df['predictions'] & df['correct'] == True)
        false_positives = np.where(df['predictions'].eq(True) & df['correct'].eq(False))
        true_negatives  = np.where(df['predictions'].eq(False) & df['correct'].eq(True))
        false_negatives = np.where(df['predictions'] & df['correct'] == False)
        num_tp = len(true_positives[0])
        num_fp = len(false_positives[0])
        num_tn = len(true_negatives[0])
        num_fn = len(false_negatives[0])

        # Precision: Of all that survived, what fraction actually survived?
        precision = 0
        try:
            precision = num_tp / (num_tp + num_fp)
        except ZeroDivisionError:
            print('Cant measure precision because it predicted no True Poss and False Poss')

        # Recall: of all that survived, what fraction did this predict as surviving?
        recall = 0
        try:
            recall = num_tp / (num_tp + num_fp)
        except ZeroDivisionError:
            print('Cant measure recall because it predicted no True pos and False neg')

        error_metrics  = '''
            True Positives: {}
            False Positives: {}
            True Negatives: {}
            False Negatives: {}
            Recall: {}
            Precision: {}
        '''.format(num_tp, num_fp, num_tn, num_fn, recall, precision)
        print(error_metrics)

        # Save error metrics in a text file
        filename = "error_metrics/errors-metrics-{}.txt".format(time.strftime("%Y-%m-%d-%H%M"))
        text_file = open(filename, "w")
        text_file.write(error_metrics)
        text_file.close()


    # Prediction Analysis
    print('PREDICTIONS: ', dict(zip(unique, counts)))



    return h2 > threshold
