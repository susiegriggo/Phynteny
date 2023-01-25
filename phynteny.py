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
parser.add_argument('-m', '--model', action='store', help='Path to custom LSTM model', deafult='nothing')
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

predict.predict(data,) #how to instantiate the default model

# write output to a genbank file
