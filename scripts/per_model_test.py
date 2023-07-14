"""
Evaluate all folds of a model and compute the phynteny score
"""

# imports
from phynteny_utils import statistics
from sklearn.metrics import classification_report
import numpy as np
import glob
import pickle5
import click
import tensorflow as tf
import pkg_resources


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

    print("Getting utils...", flush = True)

    # fetch the different PHROG categories
    category_path = pkg_resources.resource_filename(
        "phynteny_utils", "phrog_annotation_info/integer_category.pkl"
    )
    category_names = pickle5.load(open(category_path, "rb"))

    # import the confidence dict
    confidence_path = pkg_resources.resource_filename(
        "phynteny_utils", "phrog_annotation_info/confidence_kde.pkl"
    )
    confidence_dict = pickle5.load(open(confidence_path, "rb"))

    # fetch the testing data
    print('reading in data', flush = True)
    test_X = pickle5.load(open(x, "rb"))
    test_X = list(test_X.values())
    test_y = pickle5.load(open(y, "rb"))
    test_y = list(test_y.values())

    # fetch the models
    print('getting the models',flush = True)
    files = glob.glob(base + "/*")
    models = [tf.keras.models.load_model(f) for f in files]

    # compute the phynteny scores
    print("Computing Phynteny Scores", flush = True)
    scores = statistics.phynteny_score(test_X, len(category_names), models)

    # compute the confidence scores using the computed kernel densities
    print('Computing confidence', flush = True)
    known_categories = statistics.known_category(test_X, test_y, len(category_names))
    predictions_out, confidence_out = statistics.compute_confidence(
        scores, confidence_dict, category_names
    )
    predictions_df = pd.DataFrame(
        {'true_category': known_categories, 'predicted_category': predictions_out,
         'phynteny_score': [np.max(i) for i in scores], 'confidence': confidence_out})

    print('Saving the predictions to a file', flush = True)
    predictions_df.to_csv(out + "_predictions.tsv", sep='\t')

    # Build the ROC curve based on the Phynteny scores
    # Would this be any different based on the transformed confidence (probs not)
    #known_categories = statistics.known_category(test_X, test_y, len(category_names))
    #print("Building ROC curve")
    #ROC_df = statistics.build_roc(scores, known_categories, category_names)
    #ROC_df.to_csv(out + "ROC.tsv", sep="\t")

    # compute the classification report
    #print("Generating metrics")
    # note that this does not use confidence
    #report = classification_report(
    #    known_categories, [np.argmax(i) for i in scores], output_dict=True
    #)
    #with open(out + "report.pkl", "wb") as f:
    #    pickle5.dump(report, f)

    # calculate the AUC for each category
    #print("Calculating AUC")
    #auc = statistics.per_category_auc(scores, known_categories, category_names)
    #with open(out + "AUC.pkl", "wb") as f:
    #    pickle5.dump(auc, f)

    # get the thresholds
    #print("Generating thresholds")
    #phynteny_df = statistics.threshold_metrics(scores, known_categories, category_names)
    #phynteny_df.to_csv(out + "phynteny_score_metrics.tsv", sep="\t")

    # repeat on the confidence scores
    # get the thresholds
    #print("Comparing confidence")
    #phynteny_df = statistics.confidence_metrics(
    #    scores, confidence_out, known_categories, category_names
    #)
    #phynteny_df.to_csv(out + "confidence_metrics.tsv", sep="\t")

    print("FINISHED")


if __name__ == "__main__":
    main()
