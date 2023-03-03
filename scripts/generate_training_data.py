import pickle
import re
import click
from phynteny_utils import handle_genbank
from phynteny_utils import format_data
from sklearn.model_selection import train_test_split
import pkg_resources
import numpy as np

@click.command()
@click.option(
    "-i",
    "--input_data",
    help="Text file containing genbank files to build model",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-g",
    "--maximum_genes",
    type=int,
    help="Specify the maximum number of genes in each genome",
    default=120,
)
@click.option(
    "-c",
    "--gene_categories",
    type=int,
    help="Specify the minimum number of cateogries in each genome",
    default=4,
)
@click.option(
    "--prefix",
    "-p",
    default='data',
    type=str,
    help="Prefix for the output files",
)


def get_data(input_data, gene_categories, phrog_integer, maximum_genes):
    """
    Loop to fetch training and test data

    :param input_data: path to where the input files are located
    :param gene_categories: number of gene categories which must be included
    :return: curated data dictionary
    """

    training_data = {}  # dictionary to store all of the training data

    prophage_counter = 0  # count the number of prophages encountered
    prophage_pass = 0  # number of prophages which pass the filtering steps

    with open(input_data, "r") as file:

        genbank_files = file.readlines()

        for genbank in genbank_files:

            # convert genbank to a dictionary
            gb_dict = handle_genbank.get_genbank(genbank)
            gb_keys = list(gb_dict.keys())

            for key in gb_keys:

                # update the counter
                prophage_counter += 1

                # extract the relevant features
                phage_dict = handle_genbank.extract_features(gb_dict.get(key))

                # integer encoding of phrog categories
                integer = handle_genbank.phrog_to_integer(
                    phage_dict.get("phrogs"), phrog_integer
                )
                phage_dict["categories"] = integer

                # evaluate the number of categories present in the phage
                categories_present = set(integer)
                if 0 in categories_present:
                    categories_present.remove(0)

                # if above the minimum number of categories are included
                if (
                        len(phage_dict.get("phrogs")) <= maximum_genes
                        and len(categories_present) >= gene_categories
                ):
                    # update the passing candidature
                    prophage_pass += 1

                    # update dictionary with this entry
                    g = re.split(",|\.", re.split("/", genbank.strip())[-1])[0]
                    training_data[g + "_" + key] = phage_dict

    return training_data

def test_train(data, path, num_functions, max_genes = 120, test_size = 11):
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
    X, y = format_data.generate_dataset(data, 'all', num_functions, max_genes)
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

def main(input_data, maximum_genes, gene_categories, prefix):

    print("STARTING")

    # read in annotations
    phrog_integer = pkg_resources.resource_filename('phynteny_utils', 'phrog_annotation_info/phrog_integer.pkl')
    phrog_integer["No_PHROG"] = 0
    num_functions = len(
            list(set(phrog_integer.values()))
        )

    # takes a text file where each line is the file path to genbank files of phages to train a model
    print("getting input", flush=True)
    print(input, flush=True)
    data = get_data(input_data, gene_categories, phrog_integer, maximum_genes)  # dictionary to store all of the training data

    # save the training data dictionary
    print("Done Processing!\n")
    print("Removing duplicate phrog category orders")
    
    # dereplicate the data and shuffle 
    derep_dict= handle_genbank.derep_trainingdata(data)
    
    # save the original data
    with open(prefix + "_all_data.pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close() 

    # save the de-replicated data
    with open(prefix + "_dereplicated.pkl", "wb") as handle:
        pickle.dump(derep_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    # save the testing and training datasets
    test_train(data, prefix, num_functions, maximum_genes)

if __name__ == "__main__":
    main()
