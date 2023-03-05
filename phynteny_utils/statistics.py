# imports
import pickle
from phynteny_utils import format_data
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def cutoff_youdens_j(fpr, tpr, thresholds):
    """
    Calculate Youdens J for selecting the optimal threshold for predictions

    :param fpr: false positive rate
    :param tpr: true positive rate
    :param thresholds: thresholds used for calculating true positives and false positives
    :return: Youdens index (integer value)
    """
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]


def generate_test_instance(
    encoding, feature_list, num_functions, n_features, max_length
):
    """
    Mask a a function to make predictions
    """

    idx = random.randint(1, len(encoding) - 1)  # don't include ends

    # make sure that the mask is not an uknown category
    while encoding[idx] == 0:
        idx = random.randint(1, len(encoding) - 1)

    X, y = format_data.generate_example(
        encoding, feature_list, num_functions, n_features, max_length, idx
    )

    return X, y, idx


def prepare_test_data(encodings, features, num_functions, n_features, max_length):
    """
    Generate a set of predictions to evaluate the model
    """

    X_list = []
    y_list = []
    idx_list = []

    for i in range(0, len(encodings)):
        X, y, idx = generate_test_instance(
            encodings[i], features[i], num_functions, n_features, max_length
        )

        X_list.append(X)
        y_list.append(y)
        idx_list.append(idx)

    return X_list, y_list, idx_list


def get_predictions(
    X_list, y_list, idx_list, model, num_functions, n_features, max_length
):
    """
    For a set of encoded prophages, maska single a gene and predict its function.

    :param encodings: Prophages encoded as integer encoding of each gene
    :param features: Features corresponding to each prophage
    :param model: LSTM model to use for predictions
    :param num_functions: Number of possible functions used in encoding
    :param n_features: Number of features to use in encoding
    :param max_length: cutoff for the maximum number of genes in a prophage
    :return: the probabilities of each predictions, the predicted cateogory of each prediction, one-vs-rest category predictions, and one-vs-rest proability predictions
    """

    prob_list = [[] for i in range(num_functions - 1)]
    cat_list = [[] for i in range(num_functions - 1)]

    all_prob_list = []
    all_cat_list = []

    # for i in range(0,1000):
    for i in range(0, len(idx_list)):
        if i % 1000 == 0:
            print(
                str(round(i * 100 / len(idx_list), 2)) + "% of the way through",
                flush=True,
            )

        yhat = model.predict(X_list[i], verbose=0)

        # rescale so that the unknown probabilities are not included
        softmax = np.zeros(num_functions)
        softmax[1:] = yhat[0][idx_list[i]][1:] / np.sum(yhat[0][idx_list[i]][1:])
        # softmax = yhat[0][idx_list[i]]

        # update with the correct probability
        correct_category = np.argmax(y_list[i][0][idx_list[i]])
        cat_list[correct_category - 1].append(1)
        prob_list[correct_category - 1].append(softmax[correct_category])

        # add the 'rest'
        cat_list[correct_category - 1].append(0)
        prob_list[correct_category - 1].append(1 - softmax[correct_category])

        # append to all lists
        all_prob_list.append(
            softmax
        )  # needs to be from one onwards otherwise includes probability of being predicted to be unknown
        all_cat_list.append(correct_category)

    return all_prob_list, all_cat_list, cat_list, prob_list


def calculate_thresholds(cat_list, prob_list, categories):
    """
    Calculate cutoff for each PHROG category using Youden's J

    :param cat_list: one-vs-rest category for a set of predictions
    :param prob_list: one-vs-rest probability for a set of predictions
    :param categories: keys: keys of the corresponding categories
    :return: dictionary reporting Youden's J for each PHROG category
    """

    thresholds = []

    for i in range(len(categories)):
        fpr, tpr, threshold = roc_curve(cat_list[i], prob_list[i])
        thresholds.append(cutoff_youdens_j(fpr, tpr, threshold))

    return dict(zip(categories, thresholds))


def calculate_category_AUC(cat_list, prob_list, categories):
    """
    Calculate the AUC for each functional category.

    :param cat_list: one-vs-rest category for a set of predictions
    :param prob_list: one-vs-rest probability for a set of predictions
    :param categories: keys: keys of the corresponding categories
    :return: dictionary containing the Area Under the Curve (AUC) for each functional category
    """
    aucs = []
    for i in range(len(categories)):
        fpr, tpr, threshold = roc_curve(cat_list[i], prob_list[i])

        aucs.append(roc_auc_score(cat_list[i], prob_list[i]))

    return dict(zip(categories, aucs))


def calculate_metrics(
    all_cat_list, all_prob_list, thresholds, categories, num_functions
):
    """
    Calculate the precision, recall and f1 score for a set of category predictions

    :param all_cat_list: true category of a set of predictions
    :param all_prob_list: predicted softmax scores for a set of predictions
    :param thresholds: thresholds to use for each cateogry as determined using Youden's index
    :return: precision, recall, f1 score
    """

    # create vectors to store true_positives, false positives, true negatives and false negatives
    true_positive = np.zeros(num_functions - 1)
    false_positive = np.zeros(num_functions - 1)
    true_negative = np.zeros(num_functions - 1)
    false_negative = np.zeros(num_functions - 1)

    thresh_look = dict(
        zip([i for i in range(1, len(thresholds) + 1)], thresholds.values())
    )
    thresh_look[0] = 1  # where model predicts unknown is unknown is not considered

    for i in range(len(all_prob_list)):
        if np.max(all_prob_list[i]) > thresh_look.get(np.argmax(all_prob_list[i])):
            if all_cat_list[i] == np.argmax(all_prob_list[i]):
                true_positive[all_cat_list[i] - 1] += 1

            else:
                false_positive[
                    np.argmax(all_prob_list[i]) - 1
                ] += 1  # category predicted is a false positive
                false_negative[
                    all_cat_list[i] - 1
                ] += 1  # category which was meant to be predicted is a false negative

        else:
            false_negative[
                all_cat_list[i] - 1
            ] += 1  # actual category which is below the required threshold

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    ave_precision = np.sum(true_positive) / (
        np.sum(true_positive) + np.sum(false_positive)
    )
    ave_recall = np.sum(true_positive) / (
        np.sum(true_positive) + np.sum(false_negative)
    )
    ave_f1 = 2 * ave_precision * ave_recall / (ave_precision + ave_recall)

    precision_list = list(precision)
    precision_list.append(ave_precision)
    recall_list = list(recall)
    recall_list.append(ave_recall)
    f1_list = list(f1)
    f1_list.append(ave_f1)
    category_list = categories[:]
    category_list.append("average")

    return {
        "category": category_list,
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
    }


def error_margin(values):
    """
    Get the margin of error, to be used for calculating confidence intervals

    :param values: List of values to calculate margin of error for
    :return: margin of error
    """

    return np.std(values) / np.sqrt(len(values))


def plot_loss(history):
    """
    Construct a plot comparing the loss and validation loss to evaluate tranining.
    Prints the best validation loss and the file path of the model with the lowest validation loss

    :param history: list of the location of each of the history files
    """

    best_model = ""
    best_loss = 1

    for h in history:
        file = open(h, "rb")
        h_dict = pickle.load(file)
        file.close()

        epochs = np.array([i for i in range(1, len(h_dict.get("loss")) + 1)])

        plt.plot(epochs - 0.5, h_dict["loss"], color="blue")
        plt.plot(epochs, h_dict["val_loss"], color="red")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.ylim(0, 0.02)
        plt.legend(["train", "validation"], loc="upper right")

        if np.min(h_dict["val_loss"][1:]) < best_loss:
            best_loss = np.min(h_dict["val_loss"][1:])
            best_model = h

    plt.show()
    print("best validation loss: " + str(best_loss))
    print("best model:" + best_model)


def plot_ROC(cat_list, prob_list, one_letter, num_functions):
    """
    Plot ROC curve for each of the categories

    :param cat_list: list of categories for the predictions
    :param prob_list: list of probabilites for each category
    :param one_letter: dictionary encoding phrog categories as integers
    :param num_functions: Number of possible functions
    """

    colors = [
        "#332288",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#CC6677",
        "#882255",
        "#AA4499",
    ]
    categories = [
        dict(zip(list(one_letter.values()), list(one_letter.keys()))).get(i)
        for i in range(1, num_functions)
    ]
    threshold_list = []

    for i in range(num_functions - 1):
        fpr, tpr, threshold = roc_curve(cat_list[i], prob_list[i])
        plt.plot(fpr, tpr, color=colors[i], label=categories[i])

    leg = plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.plot([0, 1], [0, 1], color="black", lw=1.2, linestyle="--")

    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.show()
