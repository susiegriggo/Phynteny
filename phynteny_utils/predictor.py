"""
Module to create a predictor object
"""

# imports
import tensorflow as tf
import pickle
from phynteny_utils import format_data
import numpy as np
import glob
from phynteny_utils import statistics


def get_dict(dict_path):
    """
    Helper function to import dictionaries
    """

    with open(dict_path, "rb") as handle:
        dictionary = pickle.load(handle)
    handle.close()

    return dictionary

def get_models(models):
    """
    Get the parameters for the model

    :param models: path to the directory containing the models
    """

    files = glob.glob(models + '/*')
    models = [tf.keras.models.load_model(f) for f in files]

    return models

class Predictor:
    def __init__(
        self, models, phrog_categories_path, thresholds_path, category_names_path
    ):
        self.models = get_models(models)
        self.n_features = (
            self.models[0].get_config()
            .get("layers")[0]
            .get("config")
            .get("batch_input_shape")[2]
        )
        self.max_length = (
            self.models[0].get_config()
            .get("layers")[0]
            .get("config")
            .get("batch_input_shape")[1]
        )
        self.phrog_categories = get_dict(phrog_categories_path)
        self.thresholds = get_dict(thresholds_path)
        self.category_names = get_dict(category_names_path)
        self.num_functions = len(self.category_names)

    def predict_annotations(self, phage_dict):
        """
        predict phage annotations
        """

        encodings = [
            [self.phrog_categories.get(p) for p in phage_dict.get(q).get("phrogs")]
            for q in list(phage_dict.keys())
        ]
        features = [
            format_data.get_features(phage_dict.get(q), features_included="all")
            for q in list(phage_dict.keys())
        ]

        # get the index of the unknowns
        unk_idx = [i for i, x in enumerate(encodings[0]) if x == 0]

        if len(unk_idx) == 0:
            print(
                "Your phage "
                + str(list(phage_dict.keys())[0])
                + "is already completely annotated!"
            )

        phynteny = []

        # mask each unknown function
        for i in range(len(encodings[0])):
            if i in unk_idx:
                # encode for the missing function
                X = format_data.generate_prediction(
                    encodings,
                    features,
                    self.num_functions,
                    self.n_features,
                    self.max_length,
                    i,
                )

                # predict the missing function
                #TODO introduce the phynteny score here and compare with the threshold
                scores = statistics.phynteny_score(X, self.num_functions, self.models)


                #original in this block
                #yhat = self.models.predict(X, verbose=False)

                label = self.get_best_prediction(yhat[0][i])
                phynteny.append(label)

            else:
                phynteny.append(self.category_names.get(encodings[0][i]))

        return phynteny

    def get_best_prediction(self, v):
        """
        Get the best prediction
        """

        # remove the unknown category and take the prediction
        softmax = np.zeros(self.num_functions)
        softmax[1:] = v[1:] / np.sum(v[1:])
        prediction = np.argmax(softmax)

        # compare the prediction with the thresholds
        if np.max(softmax) > self.thresholds.get(self.category_names.get(prediction)):
            return self.category_names.get(prediction)

        #TODO change this code such that rather than using the score above the threshold use the phynteny score
        else:
            return "no prediction"

