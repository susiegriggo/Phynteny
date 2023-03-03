""" 
Functions to prepare data for training with the LSTM viral gene organisation model
"""

# imports
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def encode_strand(strand):
    """
    One hot encode the direction of each gene

    :param strand: sense encoded as a vector of 1s and 2s
    :return: one hot encoding as two separate numpy arrays
    """

    encode = np.array([2 if i == "+" else 1 for i in strand])

    strand_encode = np.array([1 if i == 1 else 0 for i in encode]), np.array(
        [1 if i == 2 else 0 for i in encode]
    )

    return strand_encode[0], strand_encode[1]


def encode_start(gene_positions):
    """
    Encode the start of each gene

    :param gene_positions: vector of the start positions for each genes
    :return: vector of encoded gene positions
    """

    start = np.array([i[0] - gene_positions[0][0] for i in gene_positions])

    return np.round(start / np.max(start), 3)


def encode_intergenic(gene_positions):
    """
    generate a vector which encodes the intergenic distance between genes

    :param gene_positions: position of the gene along the genome
    :return: vector of intergenic gaps
    """

    intergenic = [
        gene_positions[i + 1][0] - gene_positions[i][1]
        for i in range(len(gene_positions) - 1)
    ]

    intergenic.insert(0, 0)

    return np.array(intergenic)


def count_direction(strand):
    """
    Count the number of genes which have occured with the same orientation

    :param sense: Determine the number of sequences which have occured in the same direction.
    :return: List of counts of genes occuring with the same orientation
    """

    direction_count = []
    counter = 0

    for i in range(len(strand) - 1):
        if strand[i] == strand[i + 1]:
            counter += 1

        else:
            counter = 0

        direction_count.append(counter)

    direction_count.insert(0, 0)

    return np.array(direction_count)


def get_features(phage, features_included="all"):
    """
    Write a function which gets the features for a prophage

    :param features_included: string describing the features to be included
    :param phage: phage data object
    :return: array of features for the prophage
    """

    features = []

    # strand features
    if features_included in ["all", "strand"]:
        strand1, strand2 = encode_strand(phage.get("sense"))
        features.append(strand1)
        features.append(strand2)

    # gene start position #TODO manipulate this feature - might be able to make it an int value instead
    if features_included in ["all", "gene_start"]:
        start = encode_start(phage.get("position"))
        features.append(start)

    # intergenic distance
    if features_included in ["all", "intergenic"]:
        intergenic = encode_intergenic(phage.get("position"))
        features.append(intergenic)

    # gene length
    if features_included in ["all", "gene_length"]:
        length = np.array([i[1] - i[0] for i in phage.get("position")])
        features.append(length)

    # orientation count
    if features_included in ["all", "orientation_count"]:
        orientation_count = count_direction(phage.get("sense"))
        features.append(orientation_count)

    return np.array(features)


def one_hot_encode(sequence, n_features):
    """
    One hot encode PHROG categories as data is cateogrical.

    :param sequence: numerical sequence of PHROG cateogories
    :param n_features: total number of features in the model
    :return: numpy array containing one hot encoding
    """

    encoding = list()
    for value in sequence:
        vector = [0 for i in range(n_features)]
        vector[value] = 1
        encoding.append(vector)

    return np.array(encoding)


def encode_feature(encoding, feature, column):
    """
    Add a feature to sequence feature matrix

    :param encoding: matrix including features for some seuqnece
    :param feature: feature to append to matrix
    :param column: column to add feature
    :return: feature matrix including the new feature
    """

    encoding = encoding.astype("float64")
    encoding[: len(feature), column] = feature

    return encoding


def one_hot_decode(encoded_seq):
    """
    Return one-hot encoding of PHROG category to its original numeral value

    :param encoded_seq: one_hot encoding of the sequence
    :return: integer encoding of the PHROG cateogries present in a sequence
    """
    return [np.argmax(vector) for vector in encoded_seq]

def generate_example(sequence, features, num_functions, n_features, max_length, idx):
    """
    Convert a sequence of PHROG functions and associated features to a supervised learning problem

    :param sequence: integer encoded list of PHROG categories in a sequence
    :param features: list of features to include in problem
    :param num_functions: number of possible PHROG categories
    :param max_length: maximum length of a sequence
    :param idx: index of the category to mask
    :return: training or test example separated as X and y matrices
    """

    # check the length of the sequence
    seq_len = len(sequence)
    if seq_len > max_length:
        ValueError('Phage contains more genes than the maximum specified!')

    # pad the sequence
    padded_sequence = pad_sequences([sequence], padding="post", maxlen=max_length)[0]

    # generate encoding
    y = np.array(one_hot_encode(padded_sequence, num_functions))
    X = np.array(one_hot_encode(padded_sequence, n_features))
    if len(features) > 0:
        for f in range(len(features)):
            X = encode_feature(X, features[f], num_functions + f)

    # replace the function encoding for the masked sequence
    X[idx, 0:num_functions] = np.zeros(num_functions)

    # reshape the matrices
    X = X.reshape((1, max_length, n_features))
    y = y.reshape((1, max_length, num_functions))

    return X, y

def generate_prediction(sequence, features, num_functions, n_features, max_length, idx):
    """
    Prepare data to predict the function of all hypothetical genes in a sequence

    :param sequence: integer enoded list of PHROG categories in a sequence
    :param features: list of features to include
    :param num_functions: number of possible PHROG categories
    :param max_length: maximum length of a sequence
    :return: encoded matrix which can be parsed to the model
    """

    # construct features
    seq_len = len(sequence)
    padded_sequence = pad_sequences(sequence, padding="post", maxlen=max_length)[0]

    X = np.array(one_hot_encode(padded_sequence, n_features))
    for f in range(len(features)):
        X = encode_feature(X, features[0][f], num_functions + f)

    # mask the unknown
    X[idx, 0:num_functions] = np.zeros(num_functions)

    return X.reshape((1, max_length, n_features))


def generate_dataset(data, features_included, num_functions, max_length):
    """
    Generate a dataset ready to parse to LSTM from a dictionary of training data

    :param data: dictionary of prophages
    :param features_included: string describing which features should be included
    :param num_functions: number of different possible functions - PHROGs has 10
    :param max_length: maximum allowable size of a prophage
    :return: dataset framed for supervised learning
    :return: list of indices masked for each instance in the dataset
    """

    # features is a list of list objects
    X = []
    y = []

    keys = list(data.keys())

    for i in range(len(keys)):

        # get the encoding
        encoding = data.get(keys[i]).get("categories")
        if len(encoding) > max_length:
            raise Exception(
                "Prophage in your data is larger than the maximum allowable length. Try using a larger maximum"
            )

        # get the features
        features = get_features(data.get(keys[i]), features_included)

        # calculate the dimension
        n_features = num_functions + len(features)

        # pick a function to mask
        idx = random.randint(1, len(encoding) - 1)

        # make sure that the mask is not an uknown category
        while encoding[idx] == 0:
            idx = random.randint(1, len(encoding) - 1)

        # generate example
        this_X, this_y = generate_example(
            encoding, features, num_functions, n_features, max_length, idx
        )

        # store the data
        X.append(this_X)
        y.append(this_y)

    # reshape the data
    X = np.array(X).reshape(len(keys), max_length, n_features)
    y = np.array(y).reshape(len(keys), max_length, num_functions)

    return X, y

def test_train():
    def test_train(data, path, num_functions, max_genes=120, test_size=11):
        """
        Split the data into testing and training datasets. Saves these datasets as dictionaries

        :param data: dictionary of curate phage data
        :param path: path prefix to save the testing and training dictionaries
        :param num_functions: number of possible cateogories in the encoding
        :param max_genes: maximum number of genes to consider in a prophage
        :param test_size: proportion of the data to be included as test data (default is 11 which indicates one eleventh of the data
        """

        # get the keys of the data
        keys = list(data.keys())

        # encode the data
        X, y = generate_dataset(data, 'all', num_functions, max_genes)
        X_dict = dict(zip(keys, X))
        y_dict = dict(zip(keys, y))

        # generate a list describing which categories get masked
        categories = [
            np.where(y[i, np.where(~X[i, :, 0:num_functions].any(axis=1))[0][0]] == 1)[0][0]
            for i in range(len(X))
        ]
        train_keys, test_keys, train_cat, test_cat = train_test_split(data, categories, test_size=float(1 / test_size),
                                                                      random_state=42, stratify=categories)
        # generate a dictionary of training data which can be used
        train_X_data = dict(zip(train_keys, [X_dict.get(i) for i in train_keys]))
        train_y_data = dict(zip(train_keys, [y_dict.get(i) for i in train_keys]))
        test_X_data = dict(zip(test_keys, [X_dict.get(i) for i in test_keys]))
        test_y_data = dict(zip(test_keys, [y_dict.get(i) for i in test_keys]))

        # for the test data get the entire prophages because these can be used to test annotation of the entire genome
        test_phage = dict(zip(test_keys, [data.get(i) for i in test_keys]))

        # save each of these dictionaries
        with open(path + "_train_X.pkl", "wb") as handle:
            pickle.dump(train_X_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        with open(path + "_train_y.pkl", "wb") as handle:
            pickle.dump(train_y_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        with open(path + "_test_X.pkl", "wb") as handle:
            pickle.dump(test_X_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        with open(path + "_test_y.pkl", "wb") as handle:
            pickle.dump(test_y_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        with open(path + "_test_prophages.pkl", "wb") as handle:
            pickle.dump(test_phage, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

