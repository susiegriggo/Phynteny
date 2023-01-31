"""
Module for manipulating genbank files
"""
#imports
import pandas as pd
from pandas.errors  import EmptyDataError
import re


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

