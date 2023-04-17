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
        self, models, phrog_categories_path, category_names_path, threshold = 5
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
        self.category_names = get_dict(category_names_path)
        self.num_functions = len(self.category_names)
        self.threshold = threshold

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

        #get the unknowns as an X array
        #X = [format_data.generate_prediction(
        #            encodings,
        #            features,
        #            self.num_functions,
        #            self.n_features,
        #            self.max_length,
        #            i,
        #        ) for i in unk_idx]

        # get the scores for each unknown
        #scores = statistics.phynteny_score(np.array(X).reshape(len(X), self.max_length, self.n_features), self.num_functions, self.models)

        # filter for the best score
        #predictions = [self.get_best_prediction(s) for s in scores]

        #print(encodings)
        #print(predictions)


        # mask each unknown function
        for i in range(len(encodings[0])):

            if i in unk_idx:

                X = format_data.generate_prediction(
                    encodings,
                    features,
                    self.num_functions,
                    self.n_features,
                    self.max_length,
                    i,
                )


                yhat = statistics.phynteny_score(X, self.num_functions, self.models)

                print(yhat)
                #original in this block
                #yhat = self.models.predict(X, verbose=False)

               # label = predictions[np.where(np.array(unk_idx) == i)[0][0]]
                label = np.argmax(yhat)
                phynteny.append(label)

            else:
                phynteny.append(self.category_names.get(encodings[0][i]))

        return phynteny

    def get_best_prediction(self, s):
        """
        Get the best prediction
        """

        # determine whether best prediction fits the most likely category
        if np.max(s) >= self.threshold:

            # fetch the category for the prediction
            prediction = np.argmax(s)
            return self.category_names.get(prediction)

        # if it does not exceed the threshold then don't make a prediction
        else:

            return "no prediction"


