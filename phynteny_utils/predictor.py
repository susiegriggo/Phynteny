"""
Module to create a predictor object

Is an option to parse an custom dictionary for the confidence kde if the user trains their own model
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
    files = glob.glob(models + '/*')
    print(files) 
    return [tf.keras.models.load_model(m) for m in files if 'h5' in m]

class Predictor:
    def __init__(
        self, models, phrog_categories_path, confidence_dict, category_names_path
    ):
        self.models = get_models(models)
        self.max_length = (
            self.models[0].get_config()
            .get("layers")[0]
            .get("config")
            .get("batch_input_shape")[1]
        )
        self.phrog_categories = get_dict(phrog_categories_path)
        self.confidence_dict = get_dict(confidence_dict)
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
         
        if len(unk_idx) == 0:
            print(
                "Your phage "
                + str(list(phage_dict.keys())[0])
                + " is already completely annotated!"
            )

            phynteny = [self.category_names.get(e) for e in encodings[0]]
            predictions = [] 
            scores = []
            confidence = [] 

        elif len(encodings[0]) > 120: 
            print(
                "Your phage "
                + str(list(phage_dict.keys())[0])
                + " has more genes than the maximum of 120!"
            )
        
            phynteny = [self.category_names.get(e) for e in encodings[0]]
            predictions = []
            scores = [] 
            confidence = [] 

        else:

            # make data with the categories masked
            X = [format_data.generate_prediction(
                encodings,
                self.num_functions,
                self.max_length,
                i,
            ) for i in unk_idx]
            
            print(self.max_length) 
            print(self.num_functions)

            yhat = statistics.phynteny_score(np.array(X).reshape(len(X), self.max_length, self.num_functions), self.num_functions, self.models)
            
            scores = [yhat[i] for i in range(len(unk_idx))]


            # TODO change this so that we are not using this apporach
            # Refers to the entire block of code below
            # Need to write in way of getting the confidence dict or parsing it in to the model
             
            predictions, confidence = statistics.compute_confidence(scores, self.confidence_dict, self.category_names)

            #predictions = [self.get_best_prediction(s) for s in scores]
            #print('FOUND ' + str(len([i for i in predictions if i != 0])) + ' missing annotation(s)!')
            #encodings = np.array(encodings)
            #encodings[:, unk_idx] = predictions
            #phynteny = [self.category_names.get(e) for e in encodings[0]]
            #confidence = []


        return unk_idx, predictions, scores, confidence

    def get_best_prediction(self, s):
         """
         Updated procedure for fetching the prediction
         """

        # compute the phynteny score and confidence and return


    def get_best_prediction(self, s):
        """
        Evaluate the best category using the pre-computed thresholds
        """

        if max(s) >= self.eval_thresh.get(np.argmax(s)):
            return np.argmax(s)

        else:
            return 0
