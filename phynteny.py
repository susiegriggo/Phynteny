#!/usr/bin/env python3

"""
Phynteny: synteny-based annotation of phage genes
"""

import argparse
import sys
import numpy as np 
from phynteny import handle_genbank
from phynteny import format_data
from phynteny import predict
from argparse import RawTextHelpFormatter
from Bio import SeqIO
import pickle 
import tensorflow as tf 

__author__ = "Susanna Grigson"
__maintainer__ = "Susanna Grigson"
__license__ = "MIT"
__version__ = "0"
__email__ = "susie.grigson@gmail.com"
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
category_encoding  = {4: 'DNA, RNA and nucleotide metabolism',
              2: 'connector',
            3: 'head and packaging',
            1: 'integration and excision',
            5: 'lysis',
            6: 'moron, auxiliary metabolic gene and host takeover',
            7: 'other',
            8: 'tail',
            9: 'transcription regulation',
            0: 'unknown function'}

#TODO decide what to with the hardcoded variables
NUM_FUNCTIONS = 10 
N_FEATURES = 15   
MAX_LENGTH = 120 

# loop through the phages in the genbank file
keys = list(gb_dict.keys())

# read in PHROG categories 
with open('phrog_annotation_info/phrog_integer.pkl', 'rb') as handle: 
    categories = pickle.load(handle)  
handle.close() 
categories[0] = 0 

#load the LSTM model 
model = tf.keras.models.load_model(args.model)

#load in the thresholds 
with open(args.thresholds, 'rb') as handle: 
    thresholds = pickle.load(handle)
handle.close()  

for key in keys:
    
    phages = {} 
    phages[key] =  handle_genbank.extract_features(gb_dict.get(key)) 
    
    phages[key]['phrogs'] = [0 if i == 'No_PHROG' else int(i) for i in phages[key]['phrogs']] 
    encodings, features  = format_data.format_data(phages, categories)
     
    #how many unknowns are there in this phage 
    unk_idx = [i for i, x in enumerate(encodings[0]) if x == 0] 
     
    if len(unk_idx) == 0: 
        print('Your phage ' + str(key) + 'is already completley annotated') 

    else: 
       
        phynteny = [] 

        #mask each unknown function 
        for i in range(len(encodings[0])):

            if i in unk_idx: 
            
                X = format_data.generate_prediction(encodings, features, NUM_FUNCTIONS, N_FEATURES, MAX_LENGTH, i) 
            
                #predict the missing function 
                yhat = model.predict(X, verbose = False) 
            
                #remove the unknown category and take the prediction 
                softmax = np.zeros(NUM_FUNCTIONS)
                softmax[1:] = yhat[0][i][1:]/np.sum(yhat[0][i][1:]) 
                prediction = np.argmax(softmax)

                #compare the prediction with the thresholds 
                if np.max(softmax) > thresholds.get(category_encoding.get(prediction)):             
                    phynteny.append(category_encoding.get(prediction)) 

                else: 
                    phynteny.append('no prediction') 

            else: 

                phynteny.append(category_encoding.get(encodings[0][i]))

    #store the phynteny annotations 
    phages[key]['phynteny'] = phynteny 


#TODO write the updated genbank to a file 

# open the genbank file to write to
with open(args.outfile, 'wt') as handle:

    for key in keys:

        # extract a single phage
        this_phage = gb_dict.get(key)

        #add a step to extract phrogs here 

        #format the genbank file - file to make predictions
        #extracted_features = handle_genbank.extract_features(this_phage) #TODO find other way to arrange this information
        #encoding, features = format_data.format_data(this_phage, one_letter)
        
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
