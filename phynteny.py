#!/usr/bin/env python3

"""
Phynteny: synteny-based annotation of phage genes
"""
import argparse
import sys
from phynteny import handle_genbank
from phynteny import format_data
from phynteny import predict
from argparse import RawTextHelpFormatter
from Bio import SeqIO

__author__ = "Susanna Grigson"
__maintainer__ = "Susanna Grigson"
__license__ = "MIT"
__version__ = "0"
__email__ = "susie.grigson@flinders.edu.au"
__status__ = "development"

parser = argparse.ArgumentParser(description='Phynteny: synteny-based annotation of phage genes',
                                 formatter_class=RawTextHelpFormatter)

parser.add_argument('infile', help='input file in genbank format')
parser.add_argument('-o', '--outfile', action='store', default=sys.stdout, type=str,
                    help='where to write the output genbank file')
parser.add_argument('-m', '--model', action='store', help='Path to custom LSTM model',
                    default='model/all_chunk_trained_LSTMbest_val_loss.h5')
parser.add_argument('-t', '--thresholds', action='store', help='Path to dictionaries for a custom LSTM model',
                    default='model/all_chunk_trained_LSTMbest_val_loss_thresholds.pkl')
parser.add_argument('-V', '--version', action='version', version=__version__)

args = parser.parse_args()

gb_dict = handle_genbank.get_genbank(args.infile)
if not gb_dict:
    sys.stdout.write("Error: no sequences found in genbank file\n")
    sys.exit()

# Run Phynteny
# ---------------------------------------------------
one_letter = {'DNA, RNA and nucleotide metabolism': 4,
                  'connector': 2,
                  'head and packaging': 3,
                  'integration and excision': 1,
                  'lysis': 5,
                  'moron, auxiliary metabolic gene and host takeover': 6,
                  'other': 7,
                  'tail': 8,
                  'transcription regulation': 9,
                  'unknown function': 0}

# loop through the phages in the genbank file
keys = list(gb_dict.keys())

# open the genbank file to write to
with open(args.outfile, 'wt') as handle:

    for key in keys:

        # extract a single phage
        this_phage = gb_dict.get(key)

        #add a step to extract phrogs here 

        # format the genbank file - file to make predictions
        extracted_features = handle_genbank.extract_features(this_phage) #TODO find other way to arrange this information
        encoding, features = format_data.format_data(this_phage, one_letter)
        """
        data = format_data.generate_prediction(extracted_features.get('phrogs'), extracted_features, 10, 15, 120)

        # use lstm model to make some predictions
        yhat = predict.predict(data, args['model'])

        # use thresholds to determine which predictions to keep
        predictions = predict.threshold(yhat, args['thresholds'])

        # update the existing phage with the predictions
        out_dict = handle_genbank.add_predictions(this_phage, predictions)

        # write output to a genbank file
        SeqIO.write(out_dict, handle, 'genbank')
        #handle_genbank.write_genbank(out_dict, args['outfile'])
        """

handle.close()
