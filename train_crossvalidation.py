""" 
Train LSTM model with 10-fold cross validation 
"""

#imports
import format_data
import train_model
import pickle
import pandas as pd
import numpy as np
import argparse


def parse_args():
    """ 
    Parse args provided by the user 
    
    :return: parsed arguments 
    """
    
    parser = argparse.ArgumentParser(description='Train LSTM on PHROG orders')
    parser.add_argument('-features', '--features', help = 'Number of features to include')
    parser.add_argument('-base', '--base_file', help = 'Prefix used for the input files', required = True)
    parser.add_argument('-num_genes', '--num_genes', help = 'Maximum number of genes considered in a training instance. Genomes with a number of genes above this will not be included', default = 120, type = int)  
    parser.add_argument('-k', '--k', help = 'Value to use for cross validation', default = 10, type = int) 
    parser.add_argument('-m', '--memory_cells', help = 'Number of memory cells to use', type = int, default = 20) 
    parser.add_argument('-b', '--batch_size', help = 'Batch size', type = int, default = 128) 
    parser.add_argument('-phrogs', '--phrog_annotations', help = 'csv file containing the annotation and category of each phrog', required = True) 
    parser.add_argument('-e', '--epochs', help = 'Number of epochs', type = int, default = 120) 
    parser.add_argument('-dropout', '--dropout', help = 'Dropout for LSTM', type = float, default = 0.2) 
    parser.add_argument('-recurrent', '--recurrent_dropout', help = 'Recurrent dropout for LSTM', type = float, default = 0) 
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate for the Adam optimizer', type = float, default = 0.001) 
    parser.add_argument('-p', '--patience', help = 'Early stopping condition patience', type = int, default = 3) 
    parser.add_argument('-d', '--min_delta', help = 'Early stopping condition min delta', type = float, default = 1e-5)
    parser.add_argument('-out', '--out_file_prefix', help = 'Prefix used for the output files', required = True)
    
    return vars(parser.parse_args())


def main(): 
    
    #get arguments 
    args = parse_args() 
    
    print('loading phrogs') 
    #get phrog annotations 
    annot = pd.read_csv(args['phrog_annotations'], sep = '\t')
    cat_dict = dict(zip([str(i) for i in annot['phrog']], annot['category']))
    cat_dict[None] = 'unknown function'

    #hard-codedn dictionary matching the PHROG cateogories to an integer value 
    one_letter = {'DNA, RNA and nucleotide metabolism' : 4,
         'connector' : 2,
         'head and packaging' : 3,
         'integration and excision': 1,
         'lysis' : 5,
         'moron, auxiliary metabolic gene and host takeover' : 6,
         'other' : 7,
         'tail' : 8,
         'transcription regulation' : 9,
         'unknown function' :  0}

    #use this dictionary to generate an encoding of each phrog
    phrog_encoding = dict(zip([str(i) for i in annot['phrog']], [one_letter.get(c) for c in annot['category']]))

    #add a None object to this dictionary which is consist with the unknown 
    phrog_encoding[None] = one_letter.get('unknown function') 
        
    num_functions = len(one_letter)
    n_features = num_functions 
    
    if args['features'] == 'all': 
        n_features = num_functions + 5
        
    elif args['features'] == 'strand':
        n_features = num_functions + 2
        
    print('finished set up', flush = True) 
    train_model.train_kfold(args['base_file'], 
                            phrog_encoding, 
                            args['k'], 
                            num_functions, 
                            n_features, 
                            args['num_genes'], 
                            args['out_file_prefix'],  
                            args['memory_cells'], 
                            args['batch_size'], 
                            args['epochs'], 
                            args['dropout'], 
                            args['recurrent_dropout'], 
                            args['learning_rate'], 
                            args['patience'], 
                            args['min_delta'], 
                            args['features'])
    
if __name__ == "__main__":
    main()

