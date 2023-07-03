""" 
Module to handle statistics for Phynteny

This module uses code snippets from PHaNNs https://github.com/Adrian-Cantu/PhANNs
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, roc_curve


def phynteny_score(X_encodings, num_categories, models):
    """
    calculate the phynteny score. Tests multiple at once as it takes effort to read in the 10 models

    :param encoding: list of encoding matices to generate prediction
    :param models: list of models which have already been read in
    :return: per-class phynteny score for the test instance
    """

    # obtain the yhat values
    scores_list = [
        predict_softmax(X_encodings, num_categories, models[i])
        for i in range(len(models))
    ]

    return np.array(scores_list).sum(axis=0)


def build_confidence_dict(label, prediction, scores, bandwidth, categories):
    # range over values to compute kernel density over
    vals = np.arange(1.5, 10, 0.001)

    # save a dictionary which contains all the information required to compute confidence scores
    confidence_dict = dict()

    # loop through the categories
    print("Computing kernel denisty for each category...")
    for cat in range(1, 10):
        print("processing " + str(cat))
        # fetch the true labels of the predictions of this category
        this_labels = label[prediction == cat]

        # fetch the scores associated with these predictions
        this_scores = scores[prediction == cat]

        # separate false positives and true positives
        TP_scores = this_scores[this_labels == cat]
        FP_scores = this_scores[this_labels != cat]

        # loop through potential bandwidths
        for b in bandwidth:
            # compute the kernel density
            kde_TP = KernelDensity(kernel="gaussian", bandwidth=b)
            kde_TP.fit(TP_scores[:, cat].reshape(-1, 1))
            e_TP = np.exp(kde_TP.score_samples(vals.reshape(-1, 1)))

            kde_FP = KernelDensity(kernel="gaussian", bandwidth=b)
            kde_FP.fit(FP_scores[:, cat].reshape(-1, 1))
            e_FP = np.exp(kde_FP.score_samples(vals.reshape(-1, 1)))

            conf_kde = (e_TP * len(TP_scores)) / (
                e_TP * len(TP_scores) + e_FP * len(FP_scores)
            )

            if count_critical_points(conf_kde) <= 1:
                break

        # save the best estimators
        confidence_dict[categories.get(cat)] = {
            "kde_TP": kde_TP,
            "kde_FP": kde_FP,
            "num_TP": len(TP_scores),
            "num_FP": len(FP_scores),
            "bandwidth": b,
        }

    return confidence_dict


def known_category(X_encodings, y_encodings, num_categories):
    """
    Return the category of a masked gene in the test set

    :param X_encoding: list of encoded X instances with a single row masked
    :param y_encoding: list of encoded y instances
    :return: list of masked categories
    """

    known_category = list()

    for i in range(len(X_encodings)):
        # get the known category of the gene
        y_index = np.argmax(y_encodings[i])
        known_category.append(y_index)

    return known_category


def predict_softmax(X_encodings, num_categories, model):
    """
    Predict the function of a masked gene using a single model

    :param encoding: list of encoding matices to generate prediction
    :param model: model object
    :return: softmax prediction tensor
    """

    # obtain softmax scores for the masked genes
    X_encodings = np.array(X_encodings)
    yhat = model(X_encodings, training=False)
    scores_list = yhat

    return np.array(scores_list)


def build_roc(scores, known_categories, category_names):
    """
    Collect values to build the ROC curve

    :param scores: list of values for each category. Can be phynteny scores or softmax scores
    :param known_categories: Actual label of each sequence
    :return: dataframe for plotting the ROC curve
    """

    # normalise the scores such that ROC can be computed
    normed_scores = norm_scores(scores)
    known_categories = np.array(known_categories)

    # for output
    tpr_list = np.zeros((10001, len(category_names) - 1))
    mean_fpr = np.linspace(0, 1, 10001)

    # loop through each category
    for i in range(1, len(category_names)):
        # get items for the category in this iteration
        include = known_categories == i

        # convert the predictions to a series of binary classifications
        binary_index = [1 for j in known_categories[include]] + [
            0 for j in known_categories[~include]
        ]
        binary_scores = list(normed_scores[include][:, i - 1]) + list(
            normed_scores[~include][:, i - 1]
        )

        # compute ROC
        fpr, tpr, thresholds = roc_curve(binary_index, binary_scores)

        # store the data
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        tpr_list[:, i - 1] = tpr

    # save the curve for each category to a file
    ROC_df = pd.DataFrame(tpr_list)
    ROC_df["FPR"] = mean_fpr
    ROC_df.columns = [category_names.get(i) for i in range(len(category_names))][1:] + [
        "FPR"
    ]

    return ROC_df


def per_category_auc(scores, known_categories, category_names, method="ovr"):
    """
    Calculate the per category under the curve.
    Calculate the average AUC separately

    :param known_category: known category of each instance
    :param scores: list of either softmax or phynteny scores for each instance
    :param
    :return: AUC score for each category
    """

    # dictionary to store AUC
    auc_dict = {}

    # normalise the scores such that ROC can be computed
    normed_scores = norm_scores(scores)
    known_categories = np.array(known_categories)

    # loop through each category
    for i in range(1, len(category_names)):
        # get items for the category in this iteration
        include = known_categories == i

        # convert the predictions to a series of binary classifications
        binary_index = [1 for j in known_categories[include]] + [
            0 for j in known_categories[~include]
        ]
        binary_scores = list(normed_scores[include][:, i - 1]) + list(
            normed_scores[~include][:, i - 1]
        )

        # compute AUC
        auc_dict[category_names.get(i)] = roc_auc_score(binary_index, binary_scores)

    # calculate the average auc
    auc_dict["average"] = roc_auc_score(
        known_categories, normed_scores, multi_class="ovr"
    )

    return auc_dict


def norm_scores(scores_list):
    """
    Building a ROC curve in using sklearn requires values to add to 10 and cannot include unknown class.
    Function removes the unknown class and renormalises the output.

    :param scores_list: list of softmax or phynteny scores
    """

    return scores_list[:, 1:] / scores_list[:, 1:].sum(axis=1)[:, np.newaxis]


def get_masked(encoding, num_categories):
    """
    Get which indexes are masked in the data. Important  pre-masked testing data/

    :param encoding: encoded matrix
    :num_categories: number of gene functional categories in the encoding
    :return: list of masked indexes
    """

    return np.where(np.all(encoding[:, :num_categories] == 0, axis=1))[0][0]


def class_scores(tt, scores, is_real, prot_class, df):
    """
    Function for scoring quality of predictions and geting metrics
    Modified from PhANNs https://github.com/Adrian-Cantu/PhANNs/blob/master/model_training/08_graph.py

    :param tt: threshold cutoff to apply
    :param is_real:
    :param prot_class: cateogory to predict from
    :param df: dataframe to append to
    """

    is_predicted = [x >= tt - 0.05 for x in scores]

    # TODO check that this here is correct
    support = len(is_predicted)

    TP = sum(np.logical_and(is_real, is_predicted))
    FN = sum(np.logical_and(is_real, np.logical_not(is_predicted)))
    TN = sum(np.logical_and(np.logical_not(is_real), np.logical_not(is_predicted)))
    FP = sum(np.logical_and(np.logical_not(is_real), is_predicted))

    if not (TP + TN + FP + FN):
        return df

    num_pred = TP + FP

    if not num_pred:
        precision = 0

    else:
        precision = TP / num_pred

    num_rec = TP + FN

    if not num_rec:
        recall = 0

    else:
        recall = TP / num_rec

    fscore = (2 * TP) / (2 * TP + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    data_row = [prot_class, precision, recall, fscore, accuracy, tt, support]

    new_row = pd.DataFrame([data_row], columns=df.columns)
    df = pd.merge(df, new_row, how="outer")

    # df = df.append(pd.Series(data_row, index=df.columns), sort=False, ignore_index=True)

    return df


def threshold_metrics(scores, known_categories, category_names):
    """
    Calculate various metrics at different Phynteny scores
    Modified from PhANNs https://github.com/Adrian-Cantu/PhANNs/blob/master/model_training/08_graph.py

    :param scores: phytneny scores for each category
    :param known_categories: Actual label of each sequence
    :param category_names: dictionary of category labels
    """

    d = {
        "class": [],
        "precision": [],
        "recall": [],
        "f1-score": [],
        "accuracy": [],
        "threshold": [],
        "support": [],
    }

    score_range = np.arange(0, 10.1, 0.01)
    df_test_score = pd.DataFrame(data=d)

    scores_index = np.array([np.argmax(i) for i in scores])
    known_categories = np.array(known_categories)

    # loop through each category and take the predictions made to that class (regardless whether successful)
    for num in range(1, len(category_names)):
        test_set_p = scores[scores_index == num, num]
        test_set_t = known_categories[scores_index == num] == num

        for tt in score_range:
            df_test_score = class_scores(
                tt,
                np.around(test_set_p[test_set_p >= tt - 0.05], decimals=1),
                test_set_t[test_set_p >= tt - 0.05],
                num,
                df_test_score,
            )

    # TODO test the effect of removing the 0.05. Why did PHANNs include this to begin with
    df_test_score["class"] = [int(i) for i in df_test_score["class"]]
    df_test_score["category"] = [category_names.get(i) for i in df_test_score["class"]]

    return df_test_score


def confidence_metrics(scores, confidence_out, known_categories, category_names):
    """
    Calculate various metrics at different Phynteny scores
    Modified from PhANNs https://github.com/Adrian-Cantu/PhANNs/blob/master/model_training/08_graph.py

    :param scores: phynteny scores for each category
    :param confidence_out: confidence associated with the best prediction
    :param known_categories: Actual label of each sequence
    :param category_names: dictionary of category labels
    """

    d = {
        "class": [],
        "precision": [],
        "recall": [],
        "f1-score": [],
        "accuracy": [],
        "confidence": [],
        "support": [],
    }

    score_range = np.arange(0, 1.01, 0.001)
    df_test_score = pd.DataFrame(data=d)

    scores_index = np.array([np.argmax(i) for i in scores])
    known_categories = np.array(known_categories)

    # loop through each category and take the predictions made to that class (regardless whether successful)
    for num in range(1, len(category_names)):
        test_set_p = confidence_out[scores_index == num]
        test_set_t = known_categories[scores_index == num] == num


        for tt in score_range:
            df_test_score = class_scores(
                tt,
                np.around(test_set_p[test_set_p >= tt - 0.05], decimals=1),
                test_set_t[test_set_p >= tt - 0.05],
                num,
                df_test_score,
            )

    # TODO test the effect of removing the 0.05. Why did PHANNs include this to begin with
    df_test_score["class"] = [int(i) for i in df_test_score["class"]]
    df_test_score["category"] = [category_names.get(i) for i in df_test_score["class"]]

    return df_test_score


def count_critical_points(arr):
    return np.sum(np.diff(np.sign(np.diff(arr))) != 0)


def compute_confidence(scores, confidence_dict, categories):
    """
    Function which computes the confidence of a Phynteny prediction
    Input is a vector of Phynteny scores
    """

    # get the prediction for each score
    score_predictions = np.array([np.argmax(score) for idx, score in enumerate(scores)])

    # make an array to store the confidence of each prediction
    confidence_out = np.zeros(len(scores))
    predictions_out = np.zeros(len(scores))

    # loop through each of potential categories
    for i in range(1, 10):
        # get the scores relevant to the current category
        cat_scores = np.array(scores)[score_predictions == i]

        if len(cat_scores) > 0:
            # compute the kernel density estimates
            e_TP = np.exp(
                confidence_dict.get(categories.get(i))
                .get("kde_TP")
                .score_samples(cat_scores[:, i].reshape(-1, 1))
            )
            e_FP = np.exp(
                confidence_dict.get(categories.get(i))
                .get("kde_FP")
                .score_samples(cat_scores[:, i].reshape(-1, 1))
            )

            # fetch the number of TP and FP
            num_TP = confidence_dict.get(categories.get(i)).get("num_TP")
            num_FP = confidence_dict.get(categories.get(i)).get("num_FP")

            # compute the confidence scores
            conf_kde = (e_TP * num_TP) / (e_TP * num_TP + e_FP * num_FP)

            # save the scores to the output vector
            confidence_out[score_predictions == i] = conf_kde
            predictions_out[score_predictions == i] = [i for k in range(len(conf_kde))]

    return predictions_out, confidence_out
