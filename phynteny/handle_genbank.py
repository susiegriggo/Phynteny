"""
Module for manipulating genbank files
"""
# imports
import pandas as pd
from pandas.errors import EmptyDataError
import re
from Bio import SeqIO
import gzip
import pickle

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
        print('empty mmseqs file: ' + phrog_file)

    return phrog_output


def get_genbank(genbank_path):
    """
    Convert genbank file to a dictionary

    param:
    return: genbank file as a dictionary
    """

    for genbank in file:

        if genbank.strip()[-3:] == '.gz':
            try:
                with gzip.open(genbank_path.strip(), 'rt') as handle:
                    gb_dict = SeqIO.to_dict(SeqIO.parse(handle, 'gb'))
            except ValueError:
                print('ERROR: ' + genbank.strip() + ' is not a genbank file!')
                raise

        else:
            try:
                with open(genbank_path.strip(), 'rt') as handle:
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
