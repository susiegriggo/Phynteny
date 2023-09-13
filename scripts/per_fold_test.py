"""
Performance testing for each k-fold separately
"""

import glob

import click
import numpy as np
import pickle5
import pkg_resources
import tensorflow as tf
from sklearn.metrics import classification_report

# imports
from phynteny_utils import statistics


# get args
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
    category_names = pickle5.load(open(category_path, "rb"))

    # fetch the testing data
    test_X = pickle5.load(open(x, "rb"))
    test_X = list(test_X.values())
    test_y = pickle5.load(open(y, "rb"))
    test_y = list(test_y.values())

    # fetch the models
    files = glob.glob(base + "/*")
    models = [tf.keras.models.load_model(f) for f in files]

    # loop through each of the provided models
    for i in range(len(models)):
        print("Processing model " + str(i))

        # make predictions
        print("Making predictions")
        scores = statistics.predict_softmax(test_X, len(category_names), models[i])
        known_categories = statistics.known_category(
            test_X, test_y, len(category_names)
        )

        # build the PR curve for this data
        print("Building precision-recall curve")
        PR_df = statistics.build_precision_recall(scores, known_categories, category_names)
        PR_df.to_csv(out + "_" + str(i) + "_PR.tsv", sep="\t")

        # build the ROC curve for this data
        print("Building ROC curve")
        ROC_df = statistics.build_roc(scores, known_categories, category_names)
        ROC_df.to_csv(out + "_" + str(i) + "_ROC.tsv", sep="\t")

        # compute the classification report
        print("Generating report")
        report = classification_report(
            known_categories, [np.argmax(i) for i in scores], output_dict=True
        )
        with open(out + "_" + str(i) + "_report.tsv", "wb") as f:
            pickle5.dump(report, f)

        # calculate the AUC for each category
        print("Calculating AUC")
        auc = statistics.per_category_auc(scores, known_categories, category_names)
        with open(out + "_" + str(i) + "_AUC.pkl", "wb") as f:
            pickle5.dump(auc, f)

        # calculate the average precision score for each category
        print("Calculating APS")
        aps = statistics.per_category_aps(scores, known_categories, category_names)
        with open(out + "_" + str(i) + "_APS.pkl", "wb") as f:
            pickle5.dump(aps, f)

        # get the thresholds
        print("Evaluating thresholds")
        phynteny_df = statistics.threshold_metrics(
            scores, known_categories, category_names
        )
        phynteny_df.to_csv(out + "_" + str(i) + "_threshold_metrics", sep="\t")

    print("FINISHED")


if __name__ == "__main__":
    main()
