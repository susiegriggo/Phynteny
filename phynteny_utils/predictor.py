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
import sys
from loguru import logger
from phynteny_utils import statistics
from phynteny_utils import handle_genbank
from Bio import SeqIO
import click


def get_dict(dict_path):
    """
    Helper function to import dictionaries

    :param dict_path: path to the dictionary to read
    :return: dictionary object
    """

    with open(dict_path, "rb") as handle:
        dictionary = pickle.load(handle)

    handle.close()

    return dictionary


def get_models(models):
    """
    Load in genbank models

    :param models: path of directory where model obejects are located
    :return: list of models to iterate over
    """
    files = glob.glob(models + "/*")

    if len(files) == 0:
        logger.critical("Models directory is empty")
    if len(files) == 1:
        logger.warning(
            "Only one model was found. Use an ensemble of multiple models for best results."
        )
    if len([m for m in files if "h5" not in m]):
        logger.warning(
            "there are files in your models directory which are not tensorflow models"
        )

    return [tf.keras.models.load_model(m) for m in files if "h5" in m]


def run_phynteny(outfile, gene_predictor, gb_dict, categories):
    """
    Run Phynteny

    :param outfile: path to output genbank file
    :param gene_predictor: gene_predictor obejct to use for predictions
    :param gb_dict: dictionary of phages and their annotations
    :param categories: dictionary mapping PHROG categories to their corresponding integer
    :return: annotated dictionary
    """

    # get the list of phages to loop through
    keys = list(gb_dict.keys())

    # Run Phynteny
    with open(outfile, "wt") if outfile != ".gbk" else sys.stdout as handle:
        for key in keys:
            # print the phage
            print("Annotating the phage: " + key, flush=True)

            # get current phage
            phages = {key: handle_genbank.extract_features(gb_dict.get(key))}

            # get phrog annotations
            phages[key]["phrogs"] = [
                0 if i == "No_PHROG" else int(i) for i in phages[key]["phrogs"]
            ]

            # make predictions
            (
                unk_idx,
                predictions,
                scores,
                confidence,
            ) = gene_predictor.predict_annotations(phages)

            if len(predictions) > 0:
                # update with these annotations
                cds = [i for i in gb_dict.get(key).features if i.type == "CDS"]

                # return everything back
                for i in range(len(unk_idx)):
                    cds[unk_idx[i]].qualifiers["phynteny"] = categories.get(
                        predictions[i]
                    )
                    round_score = str(np.max(scores[i]))
                    cds[unk_idx[i]].qualifiers["phynteny_score"] = round_score
                    cds[unk_idx[i]].qualifiers["phynteny_confidence"] = confidence[i]

            # write to genbank file
            SeqIO.write(gb_dict.get(key), handle, "genbank")
            logger.info(f"Annotated the phage {key}")
    return gb_dict


def generate_table(outfile, gb_dict, categories, phrog_integer):
    """
    Generate table summary of the annotations made
    """

    # get the list of phages to loop through
    keys = list(gb_dict.keys())

    # count the number of genes found
    found = 0

    # convert annotations made to a text file
    with click.open_file(outfile, "wt") if outfile != ".tsv" else sys.stdout as f:
        f.write(
            "ID\tstart\tend\tstrand\tphrog_id\tphrog_category\tphynteny_category\tphynteny_score\tconfidence\tsequence\tphage\n"
        )

        for k in keys:

            # obtain the sequence
            seq = gb_dict.get(k).seq

            # get the genes
            cds = [f for f in gb_dict.get(k).features if f.type == "CDS"]

            # extract the features for the cds
            start = [c.location.start for c in cds]
            end = [c.location.end for c in cds]
            seq = [str(seq[start[i]:end[i]]) for i in range(len(cds))]

            strand = [c.strand for c in cds]
            ID = [
                c.qualifiers.get("ID")[0] if "ID" in c.qualifiers else "" for c in cds
            ]

            # lists to iterate through
            phrog = []
            phynteny_category = []
            phynteny_score = []
            phynteny_confidence = []

            # extract details for genes
            for c in cds:
                if "phrog" in c.qualifiers.keys():
                    phrog.append(c.qualifiers.get("phrog")[0])
                else:
                    phrog.append("No_PHROG")

                if "phynteny" in c.qualifiers.keys():
                    phynteny_category.append(c.qualifiers.get("phynteny"))
                    phynteny_score.append(c.qualifiers.get("phynteny_score"))
                    phynteny_confidence.append(c.qualifiers.get("phynteny_confidence"))

                    # update the number of genes found
                    if float(c.qualifiers.get("phynteny_confidence")) > 0.9:
                        found += 1

                else:
                    phynteny_category.append(np.nan)
                    phynteny_score.append(np.nan)
                    phynteny_confidence.append(np.nan)

            phrog = [int(p) if p != "No_PHROG" else p for p in phrog]
            known_category = [categories.get(phrog_integer.get(p)) for p in phrog]
            known_category = [
                "unknown function" if c == None else c for c in known_category
            ]

            # write to table
            for i in range(len(cds)):
                f.write(
                    f"{ID[i]}\t{start[i]}\t{end[i]}\t{strand[i]}\t{phrog[i]}\t{known_category[i]}\t{phynteny_category[i]}\t{phynteny_score[i]}\t{phynteny_confidence[i]}\t{seq[i]}\t{k}\n"
                )

    return found


class Predictor:
    """
    Predictor object for predicting function of unknown genes
    """

    def __init__(
        self, models, phrog_categories_path, confidence_dict, category_names_path
    ):
        self.models = get_models(models)
        self.max_length = (
            self.models[0]
            .get_config()
            .get("layers")[0]
            .get("config")
            .get("batch_input_shape")[1]
        )

        self.phrog_categories = get_dict(phrog_categories_path)
        self.confidence_dict = get_dict(confidence_dict)
        self.category_names = get_dict(category_names_path)
        self.num_functions = len(self.category_names)

    def predict_annotations(self, phage_dict):
        """ """

        encodings = [
            [self.phrog_categories.get(p) for p in phage_dict.get(q).get("phrogs")]
            for q in list(phage_dict.keys())
        ]

        if len(encodings[0]) == 0:
            logger.info(f"your phage {list(phage_dict.keys())[0]}  has zero genes!")

        unk_idx = [i for i, x in enumerate(encodings[0]) if x == 0]

        if len(unk_idx) == 0:
            logger.info(
                f"Phage {str(list(phage_dict.keys())[0])} is already completely annotated!"
            )

            predictions = []
            scores = []
            confidence = []

        elif len(encodings[0]) > 120:
            logger.info(
                f"Your phage + {str(list(phage_dict.keys())[0])} has more genes than the maximum of 120!"
            )

            predictions = []
            scores = []
            confidence = []

        else:
            # make data with the categories masked
            X = [
                format_data.generate_prediction(
                    encodings,
                    self.num_functions,
                    self.max_length,
                    i,
                )
                for i in unk_idx
            ]

            yhat = statistics.phynteny_score(
                np.array(X).reshape(len(X), self.max_length, self.num_functions),
                self.num_functions,
                self.models,
            )

            scores = [yhat[i] for i in range(len(unk_idx))]

            predictions, confidence = statistics.compute_confidence(
                scores, self.confidence_dict, self.category_names
            )

        # round the scores
        scores_round = np.round(scores, decimals=3)
        confidence_round = np.round(confidence, decimals=4)

        return unk_idx, predictions, scores_round, confidence_round
