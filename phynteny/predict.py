"""
Make predictions using LSTM model
"""

import tensorflow as tf


def predict(X, model):
    """
    Predict missing genes using LSTM model
    param X: phage data formatted into matrix
    param model: trained LSTM model
    return yhat: predicted functional categories
    """

    lstm = tf.keras.models.load_model(model)
    yhat = lstm.predict(X, verbose=0)

    return yhat

def rescale():
    """
    Rescale predictions so that unknown predictions are not included
    """

def apply_thresholds():
    """
    Apply thresholds to determine which predictions to report

    #have thresholds stored as a dictionary that are read in
    """

