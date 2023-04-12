"""
Evaluate all folds of a model and compute the phynteny score
"""

# imports
from phynteny_utils import statistics
import glob
import pickle5
import click
import tensorflow as tf
import pkg_resources

@click.command()
@click.option('--base', '-b', type=click.Path(), help='base to where the model paths are located. All k-folds should be placed in the same directory')
@click.option('-x', type=click.Path(), help='file path to the testing data X components')
@click.option('-y', type=click.Path(), help='file path to the testing data y components')
@click.option('--out', '-o', type=click.Path(), help='output file path')

def main(base, x, y, out):

    print('Getting data...')

    # fetch the different PHROG categories
    category_path = pkg_resources.resource_filename("phynteny_utils", "phrog_annotation_info/integer_category.pkl")
    category_names = pickle5.load(open(category_path, 'rb'))

    # fetch the testing data
    test_X = pickle5.load(open(x, 'rb'))
    test_X = list(test_X.values())
    test_y = pickle5.load(open(y, 'rb'))
    test_y = list(test_y.values())

    # fetch the models
    files = glob.glob(base + '/*')
    models = [tf.keras.models.load_model(f) for f in files]

    # compute the phynteny scores
    print('Computing Phynteny Scores')
    scores = statistics.phynteny_score(test_X, len(category_names), models)
    known_categories = statistics.known_category(test_X, test_y, len(category_names))

    # build the ROC curve for this data
    print('Building ROC curve')
    ROC_df = statistics.build_roc(scores, known_categories, category_names)
    ROC_df.to_csv(out + 'ROC.tsv', sep='\t')

    # compute the classification report
    print('Generating metrics')
    report = statistics.classification_report(known_categories, [np.argmax(i) for i in scores], output_dict=True)
    with open(out + 'report.tsv', "wb") as f:
        pickle5.dump(report, f)

    # calculate the AUC for each category
    print('Calculating AUC')
    auc = statistics.per_category_auc(scores, known_categories, category_names)
    with open(out + 'AUC.pkl', "wb") as f:
        pickle5.dump(auc, f)

    # get the thresholds
    print('Generating thresholds')
    phynteny_df = statistics.threshold_metrics(scores, known_categories, category_names)
    phynteny_df.to_csv(out + 'threshold_metrics', sep = '\t')

    print('FINISHED')

if __name__ == "__main__":
    main()

