""" 
Functions to prepare data for training with the LSTM viral gene organisation model
"""

# imports
import numpy as np
import random
import os
import sys 
import shutil 
import pickle5
from loguru import logger
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def instantiate_dir(output_dir, force): 
    """
    Generate output directory releative to whether force has been specififed 
    """ 

    # remove the existing outdir on force
    if force == True: 
        if os.path.isdir(output_dir) == True: 
            shutil.rmtree(output_dir)
            
        else: 
            print("\n--force was specficied even though the output directory does not exist \n")

    # make directory if force is not specified 
    else: 
        if os.path.isdir(output_dir) == True: 
            sys.exit("\nOutput directory already exists and force was not specified. Please specify -f or --force to overwrite the output directory. \n")

    # instantiate the output directory 
    os.mkdir(output_dir)


def get_dict(dict_path):
    """
    Helper function to import dictionaries

    :param dict_path: path to the desired dictionary
    :return: loaded dictionary
    """

    with open(dict_path, "rb") as handle:
        dictionary = pickle5.load(handle)
        if any(dictionary):
            logger.info(f'dictionary loaded from {dict_path}')
        else:
            logger.crtical(f'dictionary could not be loaded from {dict_path}. Is it empty?')
    handle.close()
    return dictionary

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

    :param strand: Determine the number of sequences which have occured in the same direction.
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


def one_hot_encode(sequence, num_functions):
    """
    One hot encode PHROG categories as data is cateogrical.

    :param sequence: numerical sequence of PHROG cateogories
    :param num_functions: total number of functions in the model
    :return: numpy array containing one hot encoding
    """

    encoding = list()
    for value in sequence:
        vector = [0 for i in range(num_functions)]
        vector[value] = 1
        encoding.append(vector)

    return np.array(encoding)


def one_hot_decode(encoded_seq):
    """
    Return one-hot encoding of PHROG category to its original numeral value

    :param encoded_seq: one_hot encoding of the sequence
    :return: integer encoding of the PHROG cateogries present in a sequence
    """
    return [np.argmax(vector) for vector in encoded_seq]


def generate_example(sequence, num_functions, max_length, idx):
    """
    Convert a sequence of PHROG functions and associated to a supervised learning problem

    :param sequence: integer encoded list of PHROG categories in a sequence
    :param num_functions: number of possible PHROG categories
    :param max_length: maximum length of a sequence
    :param idx: index of the category to mask
    :return: training or test example separated as X and y matrices
    """

    # check the length of the sequence
    seq_len = len(sequence)
    if seq_len > max_length:
        ValueError("Phage contains more genes than the maximum specified!")

    # pad the sequence
    padded_sequence = pad_sequences([sequence], padding="post", maxlen=max_length)[0]

    # generate encoding
    y = np.array(one_hot_encode(padded_sequence, num_functions))
    X = np.array(one_hot_encode(padded_sequence, num_functions))

    # replace the function encoding for the masked sequence
    X[idx] = np.zeros(num_functions)

    # return y just as this masked function
    y = y[idx]

    # reshape the matrices
    X = X.reshape((1, max_length, num_functions))
    y = y.reshape((1, num_functions))

    return X, y


def generate_prediction(sequence, num_functions, max_length, idx):
    """
    Prepare data to predict the function of all hypothetical genes in a sequence

    :param sequence: integer enoded list of PHROG categories in a sequence
    :param num_functions: number of possible PHROG categories
    :param max_length: maximum length of a sequence
    :return: encoded matrix which can be parsed to the model
    """

    # construct features
    seq_len = len(sequence)
    padded_sequence = pad_sequences(sequence, padding="post", maxlen=max_length)[0]

    X = np.array(one_hot_encode(padded_sequence, num_functions))

    # mask the unknown
    X[idx, 0:num_functions] = np.zeros(num_functions)

    return X.reshape((1, max_length, num_functions))


def generate_dataset(data, num_functions, max_length):
    """
    Generate a dataset ready to parse to LSTM from a dictionary of training data

    :param data: dictionary of prophages
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

        # pick a function to mask
        idx = random.randint(1, len(encoding) - 1)

        # make sure that the mask is not an uknown category
        while encoding[idx] == 0:
            idx = random.randint(1, len(encoding) - 1)

        # generate example
        this_X, this_y = generate_example(encoding, num_functions, max_length, idx)

        # store the data
        X.append(this_X)
        y.append(this_y)

    # reshape the data
    X = np.array(X).reshape(len(keys), max_length, num_functions)
    y = np.array(y).reshape(len(keys), num_functions)

    return X, y


def test_train(data, path, num_functions, max_genes=120, test_size=10):
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
    X, y = generate_dataset(data, num_functions, max_genes)
    X_dict = dict(zip(keys, X))
    y_dict = dict(zip(keys, y))

    # generate a list describing which categories get masked
    # categories = [
    #    np.where(y[i, np.where(~X[i, :, 0:num_functions].any(axis=1))[0][0]] == 1)[0][0]
    #    for i in range(len(X))
    # ]
    # masked = [statistics.get_masked(X[i], num_functions) for i in range(len(X))]
    categories = [np.where(y[i] == 1)[0][0] for i in range(len(X))]

    train_keys, test_keys, train_cat, test_cat = train_test_split(
        [i for i in range(len(categories))],
        categories,
        test_size=float(1 / test_size),
        random_state=42,
        stratify=categories,
    )

    # generate a dictionary of training data which can be used
    train_X_data = dict(
        zip([keys[i] for i in train_keys], [X_dict.get(keys[i]) for i in train_keys])
    )
    train_y_data = dict(
        zip([keys[i] for i in train_keys], [y_dict.get(keys[i]) for i in train_keys])
    )
    test_X_data = dict(
        zip([keys[i] for i in train_keys], [X_dict.get(keys[i]) for i in test_keys])
    )
    test_y_data = dict(
        zip([keys[i] for i in train_keys], [y_dict.get(keys[i]) for i in test_keys])
    )

    # for the test data get the entire prophages because these can be used to test annotation of the entire genome
    test_phage = dict(zip(test_keys, [data.get(i) for i in test_keys]))

    # save each of these dictionaries
    with open(path + "_train_X.pkl", "wb") as handle:
        pickle5.dump(train_X_data, handle)
    handle.close()
    with open(path + "_train_y.pkl", "wb") as handle:
        pickle5.dump(train_y_data, handle)
    handle.close()
    with open(path + "_test_X.pkl", "wb") as handle:
        pickle5.dump(test_X_data, handle)
    handle.close()
    with open(path + "_test_y.pkl", "wb") as handle:
        pickle5.dump(test_y_data, handle)
    handle.close()
    with open(path + "_test_prophages.pkl", "wb") as handle:
        pickle5.dump(test_phage, handle)
    handle.close()
