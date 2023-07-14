"""
Module for manipulating genbank files
"""
# imports
import pandas as pd
from pandas.errors import EmptyDataError
import re
from loguru import logger
from Bio import SeqIO
import gzip
import random
import binascii


def get_mmseqs(phrog_file):
    """
    Check that the mmseqs output exists

    param phrog_filter: location of phrog file to filter
    return: phrog annotations if they exist
    """
    try:
        phrog_output = pd.read_csv(
            phrog_file, sep="\t", compression="gzip", header=None
        )

    except EmptyDataError:
        phrog_output = pd.DataFrame()

    return phrog_output


def get_genbank(genbank):
    """
    Convert genbank file to a dictionary

    param:
    return: genbank file as a dictionary
    """

    # if genbank.strip()[-3:] == ".gz":
    if is_gzip_file(genbank.strip()):
        try:
            with gzip.open(genbank.strip(), "rt") as handle:
                gb_dict = SeqIO.to_dict(SeqIO.parse(handle, "gb"))
            handle.close()
        except ValueError:
            logger.error(genbank.strip() + " is not a genbank file!")
            raise

    else:
        try:
            with open(genbank.strip(), "rt") as handle:
                gb_dict = SeqIO.to_dict(SeqIO.parse(handle, "gb"))
            handle.close()
        except ValueError:
            logger.error(genbank.strip() + " is not a genbank file!")
            raise

    return gb_dict


def phrog_to_integer(phrog_annot, phrog_integer):
    """
    Converts phrog annotation to its integer representation
    """

    return [phrog_integer.get(i) for i in phrog_annot]


def extract_features(this_phage):
    """
    Extract the required features and format as a dictionary

    param this_phage: phage genome extracted from genbank file
    return: dictionary with the features for this specific phage
    """

    phage_length = len(this_phage.seq)
    this_CDS = [i for i in this_phage.features if i.type == "CDS"]  # coding sequences

    position = [
        (int(this_CDS[i].location.start), int(this_CDS[i].location.end))
        for i in range(len(this_CDS))
    ]
    sense = [
        re.split("]", str(this_CDS[i].location))[1][1] for i in range(len(this_CDS))
    ]
    protein_id = [
        this_CDS[i].qualifiers.get("protein_id") for i in range(len(this_CDS))
    ]
    protein_id = [p[0] if p is not None else None for p in protein_id]
    phrogs = [this_CDS[i].qualifiers.get("phrog") for i in range(len(this_CDS))]
    phrogs = ["No_PHROG" if i is None else i[0] for i in phrogs]

    return {
        "length": phage_length,
        "phrogs": phrogs,
        "protein_id": protein_id,
        "sense": sense,
        "position": position,
    }


def filter_mmseqs(phrog_output, Eval=1e-5):
    """
    Function to filter the phogs mmseqs output
    If there are two equally good annotations, the annotation with the greatest coverage is taken, then e

    param phrog_output: dataframe of phrog annotations
    param Eval: evalue to filter annotations default (1e-5)
    return: dictionary of the phrog annotations
    """

    # rename the headers
    phrog_output.columns = [
        "phrog",
        "seq",
        "alnScore",
        "seqIdentity",
        "eVal",
        "qStart",
        "qEnd",
        "qLen",
        "tStart",
        "tEnd",
        "tLen",
    ]
    phrog_output["coverage"] = phrog_output["tEnd"] - phrog_output["tStart"] + 1
    phrog_output["phrog"] = [
        re.split("_", p)[1] for p in phrog_output["phrog"]
    ]  # convert phrog to a number
    phrog_output["phrog"] = [re.split("#", p)[0][:-1] for p in phrog_output["phrog"]]

    # filter to have an e-value lower than e-5 - can change this not to be hardcoded
    phrog_output = phrog_output[phrog_output["eVal"] < Eval]

    # filter annotations with multiple hits
    phrog_output = (
        phrog_output.groupby("seq", as_index=False).coverage.max().merge(phrog_output)
    )  # hit with greatest coverage
    phrog_output = (
        phrog_output.groupby("seq", as_index=False).eVal.min().merge(phrog_output)
    )  # hit with the best evalue
    phrog_output = (
        phrog_output.groupby("seq", as_index=False).qLen.min().merge(phrog_output)
    )  # hit with the shortest query length

    return dict(zip(phrog_output["seq"].values, phrog_output["phrog"].values))


def shuffle_dict(dictionary):
    """
    Shuffles a dictionary into random order. Use to generate randomised training datasets

    :param dictionary: dictionary object to be shuffle
    :return shuffled dictionary
    """

    keys = list(dictionary.keys())
    random.shuffle(keys)

    return dict(zip(keys, [dictionary.get(key) for key in keys]))


def derep_trainingdata(training_data):
    """
    Ensure there is only one copy of each phrog order and sense order.
    Dereplication is based on the unique position and direction of genes.

    :param training_data: dictionary containing training data
    :param phrog_encoding: dictionary which converts phrogs to category integer encoding
    :return: dereplicated training dictionary
    """

    # get the training keys and encodings
    training_keys = list(training_data.keys())

    # randomly shuffle the keys
    random.shuffle(training_keys)

    # get the categories in each prophage
    training_encodings = [training_data.get(k).get("categories") for k in training_keys]

    # write a function to remove duplicates in the training data
    training_str = ["".join([str(j) for j in i]) for i in training_encodings]
    #training_sense = ["".join(training_data.get(p).get("sense")) for p in training_keys]
    #training_hash = [
    #    training_sense[i] + training_str[i] for i in range(len(training_keys))
    #]

    # get the dereplicated keys
    dedup_keys = list(dict(zip(training_str, training_keys)).values())

    return dict(zip(dedup_keys, [training_data.get(d) for d in dedup_keys]))

def add_predictions(gb_dict, predictions):
    """
    Add predictions to the genbank dictionary

    param gb_dict: genbank file as a dictionary
    param predictions: predictions to add to the genbank file
    return updated dictionary with features
    """

    keys = list(gb_dict.keys())

    for i in range(len(predictions)):
        gb_dict[keys[i]]["phynteny"] = predictions[i]
    return gb_dict


def is_gzip_file(f):
    """
    Method copied from Phispy see https://github.com/linsalrob/PhiSpy/blob/master/PhiSpyModules/helper_functions.py

    This is an elegant solution to test whether a file is gzipped by reading the first two characters.
    I also use a version of this in fastq_pair if you want a C version :)
    See https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed for inspiration
    :param f: the file to test
    :return: True if the file is gzip compressed else false
    """
    with open(f, "rb") as i:
        return binascii.hexlify(i.read(2)) == b"1f8b"


def write_genbank(gb_dict, filename):
    """
    write genbank dictionary to a file
    """

    keys = list(gb_dict.keys())

    # check for gzip
    if filename.strip()[-3:] == ".gz":
        with gzip.open(filename, "wt") as handle:
            for key in keys:
                SeqIO.write(gb_dict.get(key), handle, "genbank")
        handle.close()

    else:
        with open(filename, "wt") as handle:
            for key in keys:
                SeqIO.write(gb_dict.get(key), handle, "genbank")
        handle.close()


def get_data(input_data, gene_categories, phrog_integer, maximum_genes=False):
    """
    Loop to fetch training and test data

    :param input_data: path to where the input files are located
    :param gene_categories: number of gene categories which must be included
    :return: curated data dictionary
    """

    training_data = {}  # dictionary to store all of the training data

    prophage_counter = 0  # count the number of prophages encountered
    prophage_pass = 0  # number of prophages which pass the filtering steps

    with open(input_data, "r", errors="replace") as file:
        genbank_files = file.readlines()

        for genbank in genbank_files:
            # convert genbank to a dictionary
            gb_dict = get_genbank(genbank)
            gb_keys = list(gb_dict.keys())

            for key in gb_keys:
                # update the counter
                prophage_counter += 1

                # extract the relevant features
                phage_dict = extract_features(gb_dict.get(key))

                # integer encoding of phrog categories
                integer = phrog_to_integer(phage_dict.get("phrogs"), phrog_integer)
                phage_dict["categories"] = integer

                # evaluate the number of categories present in the phage
                categories_present = set(integer)
                if 0 in categories_present:
                    categories_present.remove(0)

                if maximum_genes == False: 
                    if len(categories_present) >= gene_categories: 

                        # update the passing candidature 
                        prophage_pass += 1 
                        
                        # update dictionary with this entry
                        g = re.split(",|\.", re.split("/", genbank.strip())[-1])[0]
                        training_data[g + "_" + key] = phage_dict 

                else: 
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
