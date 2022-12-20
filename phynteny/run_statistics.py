""" 
Compute statistics for models
"""

import format_data 
import statistics 
import pandas as pd 
import glob 
import pickle 
import tensorflow as tf 

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

#create a loop which will iterate through each of the kfold chunks 
base = '/home/grig0076/phispy_phrog_pickles/cross_validated/models/*' 
chunks = '/home/grig0076/phispy_phrog_pickles/chunks/'
all_files = glob.glob(base + '*')

best_val_loss = [f for f in all_files if 'best_val_loss.h5' in f]
all_best_val_loss = [f for f in best_val_loss if 'all' in f]

#read in the trainind data 
test_file = chunks + 'prophage_phrog_data_derep_fourormore_lessthan121_test_chunk.pkl'
test_data = pickle.load(open(test_file, 'rb')) 
test_encodings, test_features = format_data.format_data(test_data, phrog_encoding)

num_functions = len(one_letter)
n_features = num_functions + len(test_features[0]) 
max_length = 120
categories = [dict(zip(list(one_letter.values()), list(one_letter.keys()))).get(i) for i in range(1,num_functions)]

#loop through 
metric_list = [] 
support_list = [] #number in each category 
AUC_list = []  

print('going', flush = True) 

for i in range(0,len(all_best_val_loss)): 
    
    print('Validation number: ' + str(i), flush = True) 
    
    model = tf.keras.models.load_model(all_best_val_loss[i]) 
              
    all_prob_list, all_cat_list, cat_list, prob_list = statistics.get_predictions(test_encodings, test_features, model, num_functions, n_features, max_length)

    #calcualte AUC 
    print('calculating AUC', flush = True) 
    AUC = statistics.calculate_category_AUC(cat_list, prob_list, categories)
    AUC['all_ovo'] = roc_auc_score(all_cat_list, [i[1:] for i in all_prob_list], multi_class = 'ovo')
    AUC['all_ovr'] = roc_auc_score(all_cat_list, [i[1:] for i in all_prob_list], multi_class = 'ovr')
    
    #calcualte metrics 
    print('Calculating metrics', flush = True) 
    thresholds = statistics.calculate_thresholds(cat_list, prob_list, categories)
    metrics = statistics.calculate_metrics(all_cat_list, all_prob_list, thresholds, categories) 
    
    #add to the lists 
    AUC_list.append(AUC) 
    metric_list.append(metrics) 
    support_list.append(np.array([len(c) for c in cat_list])) #number in each category 
    
    print(AUC)
    
    
#Get the AUC 
AUC_df = pd.DataFrame()  
for A in AUC_list: 
    
    this_df = pd.DataFrame(A, index = [0])
    AUC_df = AUC_df.append(this_df) 
    
AUC_df.to_csv(base[:-1] + 'all_best_val_loss_AUC.tsv', sep = '\t')

#get the metrics 
crit = 1.96 #critical value for a 95%confidence interval 
df = pd.DataFrame() 

for i in range(len(metric_list)): 
    this_df = pd.DataFrame(metric_list[i]) 
    this_df['k'] = [i for j in range(len(this_df))]
    df = df.append(this_df)
    
#do stats 
precision_mean = [] 
recall_mean = [] 
f1_mean = [] 
precision_err = [] 
recall_err = [] 
f1_err = [] 

for c in categories: 
    
    values = df[df['category'] == c][['precision', 'recall', 'f1']]
    mean = np.mean(values)
    err = crit * error_margin(values)
    
    precision_mean.append(mean['precision'])
    recall_mean.append(mean['recall'])
    f1_mean.append(mean['f1'])
    
    precision_err.append(err['precision'])
    recall_err.append(err['recall'])
    f1_err.append(err['f1'])
    
metric_df = pd.DataFrame({'category': categories, 
                   'precision': precision_mean, 
                   'recall': recall_mean, 
                   'f1': f1_mean, 
                    'precision_err': precision_err, 
                    'recall_err': recall_err, 
                    'f1_err':f1_err})

metric_df.to_csv(base[:-1] + 'all_best_val_loss_AUC.tsv', sep = '\t') 