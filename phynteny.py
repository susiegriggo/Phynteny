#!/usr/bin/env python3

"""
Phynteny: synteny-based annotation of phage genes
"""
import argparse
import sys
from phynteny import handle_genbank
from phynteny import format_data
from phynteny import predict

__author__ = "Susanna Grigson"
__maintainer__ = "Susanna Grigson"
__license__ = "MIT"
__version__ = "0"
__email__ = "susie.grigson@flinders.edu.au"
__status__ = "development"

parser = argparse.ArgumentParser(description='Phynteny: synteny-based annotation of phage genes',
                                 formatter_class=RawTextHelpFormatter)

parser = argparse.add_argument('infile', type=is_valid_file, help='input file in genbank format')
parser.add_argument('-o', '--outfile', action='store', default=sys.stdout, type=argparse.FileType('w'),
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

# format the genbank file - file to make predictions
input = format_data.extract_features(gb_dict)
data = format_data.generate_prediction(input)

# use lstm model to make some predictions
yhat = predict.predict(data, args['model'])

# use thresholds to determine which predictions to keep
predictions = predict.threshold(yhat, args['thresholds'])

# update the existing phage with the predictions
out_dict = handle_genbank.add_predictions(gb_dict, predictions)

# write output to a genbank file
handle_genbank.write_genbank(out_dict, args['outfile'])
