
#!/usr/bin/python

"""
Module for manipulating genbank files
"""
# imports
import pandas as pd
from pandas.errors import EmptyDataError
import re
from Bio import SeqIO
import gzip
import random


def get_mmseqs(phrog_file):
    """
    Check that the mmseqs output exists

    param phrog_filter: location of phrog file to filter
    return: phrog annotations if they exist
    """
    try:

        phrog_output = pd.read_csv(phrog_file, sep='\t', compression='gzip', header=None)

    except EmptyDataError:

        phrog_output = pd.DataFrame()
        #print('empty mmseqs file: ' + phrog_file)

    return phrog_output


def get_genbank(genbank):
    """
    Convert genbank file to a dictionary

    param:
    return: genbank file as a dictionary
    """

    if genbank.strip()[-3:] == '.gz':
        try:
            with gzip.open(genbank.strip(), 'rt') as handle:
                gb_dict = SeqIO.to_dict(SeqIO.parse(handle, 'gb'))
        except ValueError:
            print('ERROR: ' + genbank.strip() + ' is not a genbank file!')
            raise

    else:
        try:
            with open(genbank.strip(), 'rt') as handle:
                gb_dict = SeqIO.to_dict(SeqIO.parse(handle, 'gb'))
        except ValueError:
            print('ERROR: ' + genbank.strip() + ' is not a genbank file!')
            raise

    return gb_dict


def phrog_to_integer(phrog_annot, phrog_integer):
    """
    Converts phrog annotation to its integer representation
    """
    return [phrog_integer.get(i) for i in phrog_annot]


def integer_to_category():
    """
    Converts integer encoding to its relevant phrog category
    """


def extract_features(this_phage):
    """
    Extract the required features and format as a dictionary

    param this_phage: phage genome extracted from genbank file
    return: dictionary with the features for this specific phage
    """
 
    phage_length = len(this_phage.seq)
    this_CDS = [i for i in this_phage.features if i.type == 'CDS']  # coding sequences

    position = [(int(this_CDS[i].location.start), int(this_CDS[i].location.end)) for i in range(len(this_CDS))]
    sense = [re.split(']', str(this_CDS[i].location))[1][1] for i in range(len(this_CDS))]
    protein_id = [this_CDS[i].qualifiers.get('protein_id') for i in range(len(this_CDS))]
    protein_id = [p[0] if p is not None else None for p in protein_id]
    phrogs = [this_CDS[i].qualifiers.get('phrog') for i in range(len(this_CDS))]
    phrogs = ['No_PHROG' if i is None else i[0] for i in phrogs]

    return {'length': phage_length, 'phrogs': phrogs, "protein_id": protein_id, "sense": sense,
            "position": position}


def filter_mmseqs(phrog_output, Eval=1e-5):
    """
    Function to filter the phogs mmseqs output
    If there are two equally good annotations, the annotation with the greatest coverage is taken, then e

    param phrog_output: dataframe of phrog annotations
    param Eval: evalue to filter annotations default (1e-5)
    return: dictionary of the phrog annotations
    """

    # rename the headers
    phrog_output.columns = ['phrog', 'seq', 'alnScore', 'seqIdentity', 'eVal', 'qStart', 'qEnd', 'qLen', 'tStart',
                            'tEnd', 'tLen']
    phrog_output['coverage'] = phrog_output['tEnd'] - phrog_output['tStart'] + 1
    phrog_output['phrog'] = [re.split('_', p)[1] for p in phrog_output['phrog']]  # convert phrog to a number
    phrog_output['phrog'] = [re.split('#', p)[0][:-1] for p in phrog_output['phrog']]

    # filter to have an e-value lower than e-5 - can change this not to be hardcoded
    phrog_output = phrog_output[phrog_output['eVal'] < Eval]

    # filter annotations with multiple hits
    phrog_output = phrog_output.groupby('seq', as_index=False).coverage.max().merge(
        phrog_output)  # hit with greatest coverage
    phrog_output = phrog_output.groupby('seq', as_index=False).eVal.min().merge(
        phrog_output)  # hit with the best evalue
    phrog_output = phrog_output.groupby('seq', as_index=False).qLen.min().merge(
        phrog_output)  # hit with the shortest query length

    return dict(zip(phrog_output['seq'].values, phrog_output['phrog'].values))


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
    # training_encodings = [[phrog_encoding.get(i) for i in training_data.get(key).get('phrogs')] for key in
    # training_keys]

    training_encodings = [training_data.get(k).get('categories') for k in training_keys]

    # write a function to remove duplicates in the training data
    training_str = [''.join([str(j) for j in i]) for i in training_encodings]
    training_sense = [''.join(training_data.get(p).get('sense')) for p in training_keys]
    training_hash = [training_sense[i] + training_str[i] for i in range(len(training_keys))]

    # get the dereplicated keys
    dedup_keys = list(dict(zip(training_hash, training_keys)).values())

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

def write_genbank(gb_dict, filename):
    """
    write genbank dictionary to a file
    """

    keys = list(gb_dict.keys())

    # check for gzip
    if filename.strip()[-3:] == '.gz':

        with gzip.open(filename, 'wt') as handle:
            for key in keys:
                SeqIO.write(gb_dict.get(key), handle, 'genbank')
        handle.close()

    else:

        with open(filename, 'wt') as handle:
            for key in keys:
                SeqIO.write(gb_dict.get(key), handle, 'genbank')
        handle.close()
