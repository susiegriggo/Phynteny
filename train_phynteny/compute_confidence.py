"""
Evaluate all folds of a model and compute the phynteny score
"""
import glob
import pickle

import click
import numpy as np
import pandas as pd
import pkg_resources
import tensorflow as tf

# imports
from phynteny_utils import statistics


@click.command()
@click.option(
    "--base",
    "-b",
    type=click.Path(),
    help="base to where the model paths are located. All k-folds should be placed in the same directory",
)
@click.option(
    "-x", type=click.Path(), help="file path to the testing data X components"
)
@click.option(
    "-y", type=click.Path(), help="file path to the testing data y components"
)
@click.option("--out", "-o", type=click.Path(), help="output file path")
def main(base, x, y, out):
    print("Getting data...")

    # fetch the different PHROG categories
    category_path = pkg_resources.resource_filename(
        "phynteny_utils", "phrog_annotation_info/integer_category.pkl"
    )
    category_names = pickle.load(open(category_path, "rb"))

    # fetch the testing data
    test_X = pickle.load(open(x, "rb"))
    test_X = list(test_X.values())
    test_y = pickle.load(open(y, "rb"))
    test_y = list(test_y.values())

    # fetch the models
    files = glob.glob(base + "/*")
    models = [tf.keras.models.load_model(f) for f in files]

    # compute the phynteny scores
    print("Computing Phynteny Scores")
    scores = statistics.phynteny_score(test_X, len(category_names), models)

    # get the labels and the predcted labels
    label = np.array([np.argmax(i) for i in test_y])
    prediction = np.array([np.argmax(score) for idx, score in enumerate(scores)])

    # generate dictionary to sae the confidence dictionary
    bandwidth = np.arange(0, 5, 0.005)[1:]
    confidence_dict = statistics.build_confidence_dict(
        label, prediction, scores, bandwidth, category_names
    )

    # save the confidence dictionary
    pickle.dump(confidence_dict, open(out, "wb"))

    print("FINISHED")


if __name__ == "__main__":
    main()
