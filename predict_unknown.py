"""
Predict the functions of unknown proteins - looping through unknown proteins 
"""

import pandas as pd
import pickle5 
import format_data
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='predict the function of unknown proteins')
parser.add_argument('-m','--model', help='model to predict functions', required=True)
parser.add_argument('-t','--thresholds', help='thresholds to use for each cateogory', required=True)
parser.add_argument('-d', '--data', help = 'data to predict the unknown function of', required = True)
parser.add_argument('-o', '--out', help = 'text file to save output', required = True)
                    
args = vars(parser.parse_args())

#read in the PHROG categories 
#generate dictionary 
annot = pd.read_csv('/home/grig0076/LSTMs/phrog_annot_v4.tsv', sep = '\t')

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

#read in data 
file = open(args['data'],'rb')
all_data = pickle5.load(file)
file.close()
all_keys = list(all_data.keys()) 
predict_encodings, predict_features = format_data.format_data(all_data, phrog_encoding)

#set parameters
num_functions = len(one_letter)
n_features = num_functions + len(predict_features[0]) 
max_length = 120
categories =  [dict(zip(list(one_letter.values()), list(one_letter.keys()))).get(i) for i in range(0,num_functions)]

#read in the model  
model  = tf.keras.models.load_model(args['model'])
thresholds = pickle5.load(open(args['thresholds'],'rb'))
thresholds['unknown function'] = 1
 
#loop through each genome and predict the unknown genes 
with open(args['out'], 'w') as f:
    
    for i in range(len(predict_encodings)): 
        
        if i %50 == 0: 
            print(str(i*100/len(predict_encodings)) + '% of the way through', flush = True)
            
        #get the indexes of the unknowns in this prophage 
        idx = [f for f, x in enumerate(predict_encodings[i]) if x == 0]
        
        #predict each of the unknowns 
        for j in idx: 
        
            #get the protein id corresponding to this protein 
            protein_id = all_data.get(all_keys[i]).get('protein_id')[j]
            
            X, y = format_data.generate_example(predict_encodings[i], predict_features[i], num_functions, n_features, max_length, j)
            yhat = model.predict(X,verbose = False)
                
            softmax = np.zeros(num_functions)
            softmax[1:] = yhat[0][j][1:]/np.sum(yhat[0][j][1:]) 
            prediction = np.argmax(softmax)  #best category 
            
            if protein_id != None: 
                f.write(protein_id) 
            else: 
                f.write('no_id') 
                
            f.write('\t') 
                
            #check the prediction with the threshold 
            if np.max(softmax) > thresholds.get(categories[prediction]):
    
                f.write(categories[prediction]) 
                
            #otherwise record checksum as a miss 
            else: 
                f.write('not_predicted') 
                
            f.write('\n') 
f.close()          
                    