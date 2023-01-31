"""
Compare k-fold cross validation models 

Produce figures of ROC curves, thresholds and other comparison metrics 
"""

#imports
import statistics 
import format_data 
import argparse
import pandas as pd
import pickle 
from collections import Counter 
import glob 
import numpy as np 
import tensorflow as tf

parser = argparse.ArgumentParser(description='evaluate k-fold models')
parser.add_argument('-m','--models', help='Location of the models', required=True)
parser.add_argument('-t', '--test_data', help = 'pickle file containing testing prophages', required = True)
    
args = vars(parser.parse_args())

#read encoding  
annot = pd.read_csv('/home/grig0076/LSTMs/phrog_annot_v4.tsv', sep = '\t')

#hard-coded dictionary matching the PHROG cateogories to an integer value 
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

#load in the test data 
base = args['models']
test_file = args['test_data']
test_data = pickle.load(open(test_file, 'rb')) 
test_encodings, test_features = format_data.format_data(test_data, phrog_encoding)

#read in the model files 
all_files = glob.glob(base + '*')
best_val_loss = [f for f in all_files if 'best_val_loss.h5' in f]

#set parameters used for the model
num_functions = len(one_letter)
n_features = num_functions + len(test_features[0]) 
max_length = 120
categories = [dict(zip(list(one_letter.values()), list(one_letter.keys()))).get(i) for i in range(1,num_functions)]

#generate the testing data 
test_data = pickle.load(open(test_file, 'rb')) 
test_encodings, test_features = format_data.format_data(test_data, phrog_encoding)
X_list, y_list, idx_list =  statistics.prepare_test_data(test_encodings, test_features, num_functions, n_features, max_length)

#determine the number of proteins for each phrog category 
count = Counter([test_encodings[i][idx_list[i]] for i in range(len(test_encodings[0:500]))])
count_df = pd.DataFrame.from_dict(count, orient = 'index') 
support_dict = dict(zip([categories[i -1] for i in count_df.index], count_df[0]))

#save to a pickle 
with open(base + 'support_per_PHROG_cateogry_in_test_data.pkl', 'wb') as handle: 
    pickle.dump(support_dict, handle)
handle.close()

#loop through each of the models 
mean_fpr = np.linspace(0, 1, 10001)

tpr_list = [[] for i in range(num_functions-1)] 
tmp_list = [] 

#loop through each of the trained models 
for i in range(len(best_val_loss)): 
    
    print('testing model: ' + str(best_val_loss[i]), flush = True) 
    
    model = tf.keras.models.load_model(best_val_loss[i]) 
    
    all_prob_list, all_cat_list, cat_list, prob_list = get_predictions(X_list, y_list, idx_list, model, num_functions, n_features, max_length)
    
    #generate ROC curve 
    for j in range(num_functions-1):
        fpr, tpr, threshold = roc_curve(cat_list[j], prob_list[j])
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        tpr_list[j].append(tpr) 

    print('analysing..', flush = True) 
    
    #calcualte AUC 
    AUC = calculate_category_AUC(cat_list, prob_list, categories)
    AUC['all_ovo'] = roc_auc_score(all_cat_list, [i[1:] for i in all_prob_list], multi_class = 'ovo')
    AUC['all_ovr'] = roc_auc_score(all_cat_list, [i[1:] for i in all_prob_list], multi_class = 'ovr')
    
    #calcualte metrics 
    thresholds = calculate_thresholds(cat_list, prob_list, categories)
    metrics = calculate_metrics(all_cat_list, all_prob_list, thresholds, categories) 
    
    #save AUC to a dictionary 
    AUC_file = base + 'AUC/' +  re.split('/', best_val_loss[i])[-1][:-16] + '_AUC.pkl'
    with open(AUC_file, 'wb') as handle: 
        pickle.dump(AUC, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #save thresholds to a dictionary 
    threshold_file = base +  'thresholds/' + re.split('/', best_val_loss[i])[-1][:-16] + '_thresholds.pkl'
    with open(threshold_file, 'wb') as handle: 
        pickle.dump(thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #save the metrics to a dictionary 
    metric_file = base + 'metrics/'  + re.split('/', best_val_loss[i])[-1][:-16] + '_metrics.pkl' 
    with open(metric_file, 'wb') as handle: 
        pickle.dump(metrics, handle,protocol=pickle.HIGHEST_PROTOCOL) 
        
    print('done!', flush  = True) 
    print('\n', flush = True) 
        
#make average ROC curve 
print('Generating ROC curve', flush = True)  
categories = [dict(zip(list(one_letter.values()), list(one_letter.keys()))).get(i) for i in range(1,num_functions)]
ROC_df = pd.DataFrame() 
ROC_df['FPR'] = mean_fpr

for i in range(num_functions-1):
    
    tprs = np.array(tpr_list[i][:-1])
    mean_tpr = np.mean(tprs, axis = 0) 
    std_tpr = np.std(tprs, axis = 0)  
    
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = mean_tpr - std_tpr

    ROC_df[categories[i] + ' mean'] = mean_tpr
    ROC_df[categories[i] + ' lower'] = tpr_lower
    ROC_df[categories[i] + ' upper'] = tpr_upper 
    
ROC_df.to_csv(base + 'mean_ROC_curve_per_category.csv') 
   