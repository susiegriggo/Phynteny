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
parser.add_argument('-m', '--model', action='store', help='Path to custom LSTM model', deafult="link to model")
parser.add_argument('-t', '--thresholds', action='store', help='Path to dictionaries for a custom LSTM model', default = 'link to the model thresholds')
parser.add_argument('-V', '--version', action='version', version=__version__)

args = parser.parse_args()

gb_dict = handle_genbank.get_genbank(args.infile)
if not gb_dict:
    sys.stdout.write("Error: no sequences found in genbank file\n")
    sys.exit()

# Run Phynteny
# ---------------------------------------------------

# format the genbank file - file to make predictions
this_phage = format_data.extract_features(gb_dict)
data = format_data.generate_prediction(this_phage)

# use lstm model to make some predicitions
predictions = predict.predict(data,)

# update the existing phage with the predictions


# write output to a genbank file
handle_genbank.write_genbank(this_phage, args['outfile'])
