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

def get_models(models ):
    """

    """
    print(models + '/*')
    files = glob.glob(models + '/*')


    return [tf.keras.models.load_model(m) for m in files]
class Predictor:
    def __init__(
        self, models, phrog_categories_path, threshold, category_names_path
    ):
        self.models = get_models(models)
        self.max_length = (
            self.models[0].get_config()
            .get("layers")[0]
            .get("config")
            .get("batch_input_shape")[1]
        )
        self.phrog_categories = get_dict(phrog_categories_path)
        self.threshold = threshold
        self.category_names = get_dict(category_names_path)
        self.num_functions = len(self.category_names)

    def predict_annotations(self, phage_dict):
        """

        """

        encodings = [
            [self.phrog_categories.get(p) for p in phage_dict.get(q).get("phrogs")]
            for q in list(phage_dict.keys())
        ]

        unk_idx = [i for i, x in enumerate(encodings[0]) if x == 0]
        print(len(encodings)) 
        if len(unk_idx) == 0:
            print(
                "Your phage "
                + str(list(phage_dict.keys())[0])
                + " is already completely annotated!"
            )

            phynteny = [self.category_names.get(e) for e in encodings[0]]

        elif len(encodings[0]) > 120: 
            print(
                "Your phage "
                + str(list(phage_dict.keys())[0])
                + " has more genes than the maximum of 120!"
            )
        
            phynteny = [self.category_names.get(e) for e in encodings[0]]

        else:
            # make data with the categories masked
            X = [format_data.generate_prediction(
                encodings,
                self.num_functions,
                self.max_length,
                i,
            ) for i in unk_idx]

            yhat = statistics.phynteny_score(np.array(X).reshape(len(X), self.max_length), self.num_functions, self.models)

            scores = [yhat[i] for i in range(len(unk_idx))]

            predictions = [self.get_best_prediction(s) for s in scores]
            print('FOUND ' + str(len([i for i in predictions if i != 0])) + ' missing annotation(s)!')
            encodings = np.array(encodings)
            encodings[:, unk_idx] = predictions
            phynteny = [self.category_names.get(e) for e in encodings[0]]

        return phynteny

    def get_best_prediction(self, s):

        if np.max(s) >= self.threshold:
            return np.argmax(s)

        else:
            return 0

