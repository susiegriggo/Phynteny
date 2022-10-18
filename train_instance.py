""" 
Train PHROG gene order LSTM 

This should be given just a dictionary of training data 
"""
import train_model 
import format_data
import argparse 
import pickle5
import pandas as pd
import numpy as np

def parse_args():
    """ 
    Parse args provided by the user 
    
    :return: parsed arguments 
    """
    
    parser = argparse.ArgumentParser(description='Train LSTM on PHROG orders')
    parser.add_argument('-t','--training_data', help='Training data', required=True)
    parser.add_argument('-f', '--flip_genomes', help='Flip genomes to ensure that an integrase is at the start of each sequence', required = True) 
    parser.add_argument('-m', '--memory_cells', help = 'Number of memory cells to use', required = True) 
    parser.add_argument('-b', '--batch_size', help = 'Batch size', required = True) 
    parser.add_argument('-phrogs', '--phrog_annotations', help = 'csv file containing the annotation and category of each phrog', required = True) 
    parser.add_argument('-e', '--epochs', help = 'Number of epochs', required = True) 
    parser.add_argument('-p', '--patience', help = 'Early stopping condition patience', required = True) 
    parser.add_argument('-d', '--min_delta', help = 'Early stopping condition min delta', required = True)
    parser.add_argument('-unbias', '--unbiased_categories', help = 'If True ensures that there is an equal amount of data for each PHROG category', required = True) 
    parser.add_argument('-out', '--out_file_prefix', help = 'Prefix used for the output files', required = True)
    parser.add_argument('-portion', '--training_portion', help = 'Portion of the data to use to train the model', required = True) 
    
    return vars(parser.parse_args())

def main(): 
    
    #get arguments 
    args = parse_args() 
    
    #need to read in training genomes - manipulate such that we are reading in just some consistent set of training data 
    print('opening required files', flush = True) 
    file = open(args['training_data'],'rb')
    training_data = pickle5.load(file)
    file.close()

    #generate dictionary 
    training_keys = list(training_data.keys())
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

    print('filtering data', flush = True) 
    #flip the genome if there is an integrase at the end of the sequence 
    if args['flip_genomes']: 
        training_data = format_data.flip_genomes(training_data, phrog_encoding)

    #derepicate training data 
    training_data_derep = format_data.derep_trainingdata(training_data, phrog_encoding)

    #shuffle the data 
    training_data_derep = format_data.shuffle_dict(training_data_derep)

    #generate features 
    print('generating features', flush = True) 
    training_encodings, features = format_data.format_data(training_data_derep, phrog_encoding) 
    num_functions = len(one_letter)
    max_length = np.max([len(t) for t in training_encodings])
    n_features = num_functions + len(features[0])

    #generate dataset 
    if args['unbiased_categories']: 

        print('Generating training data with unbiased cateogries', flush = True)
        
        X, y, genome_included, masked_idx = format_data.generate_dataset_unbiased_category(training_encodings, features, num_functions, n_features, max_length) 

        #split into train and test before or after generating the dataset - work on this - may not be much difference anyway 
        train_num = int(args['training_portion']*len(masked_idx)) 
        X_train = X[:train_num] 
        y_train = y[:train_num] 

        #get the test protein ids 
        genome_included_sum = np.cumsum(genome_included) 
        j = list(genome_included_sum).index(train_num)
        test_ids = training_keys[j:]

    else:
        
        print('Generating training data without unbiased cateogories', flush = True) 

        train_num = int(args['training_portion']*len(training_encodings)) 
        X_train, y_train, masked_idx = format_data.generate_dataset(training_encodings[:train_num], features[:train_num], dataset_size, num_functions, n_features, max_length) 

        #get the ids of the sequences for the test data 
        test_ids = training_keys[train_num:] 

    print('TRAINING STARTED', flush = True)
    model_file = args['out_file_prefix'] + '_trained_LSTM.md5' 
    history_file = args['out_file_prefix'] + '_history.pkl' 

    train_model.train_model(X_train, y_train, model_file, history_file, args['memory_cells'], args['batch_size'], args['epochs'], args['patience'], args['min_delta'])
    print('TRAINING COMPLETED', flush = True) 

    #Save the ids of the data in the test dataset 
    test_id_filename = args['out_file_prefix'] + '_test_prophage_ids.txt' 
    with open(test_id_filename, 'w') as f:
        for line in test_ids:
            f.write(f"{line}\n")
            
if __name__ == "__main__":
    main()