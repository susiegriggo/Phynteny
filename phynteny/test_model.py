""" 
test a single model 
""" 

#imports
import statistics 
import format_data 
import argparse
import train_model
import pandas as pd
import pickle 
from collections import Counter 
import glob 
import numpy as np 
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate model')
    parser.add_argument('-m','--model', help='Location of the model to test', required=True)
    parser.add_argument('-t', '--test_data', help = 'pickle file containing testing prophages', required = True)

    return vars(parser.parse_args())

def main(): 
    print('starting', flush = True) 
    args = parse_args() 

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
    base = args['model']
    test_file = args['test_data']
    test_data = pickle.load(open(test_file, 'rb')) 
    test_encodings, test_features = format_data.format_data(test_data, phrog_encoding)

    #set parameters used for the model
    num_functions = len(one_letter)
    n_features = num_functions #+2
    max_length = 120 
    categories = [dict(zip(list(one_letter.values()), list(one_letter.keys()))).get(i) for i in range(1,num_functions)]
    
    #select relevant features 
    test_features = train_model.select_features(test_features, 'none')
    
    #generate the testing data 
    print('N FEATURES: ' + str(n_features))  
    X_list, y_list, idx_list =  statistics.prepare_test_data(test_encodings, test_features, num_functions, n_features, max_length)

    #determine the number of proteins for each phrog category  #TODO fix names 
    count = Counter([test_encodings[i][idx_list[i]] for i in range(len(test_encodings))])
    count_df = pd.DataFrame.from_dict(count, orient = 'index') 
    support_dict = dict(zip([categories[i -1] for i in count_df.index], count_df[0]))

    #save to a pickle 
    with open( args['model'][:-4] + '_support.pkl', 'wb') as handle: 
        pickle.dump(support_dict, handle)
    handle.close()

    print('testing model: ' + args['model'], flush = True) 

    model = tf.keras.models.load_model(args['model']) 

    all_prob_list, all_cat_list, cat_list, prob_list = statistics.get_predictions(X_list, y_list, idx_list, model, num_functions, n_features, max_length)

    #generate ROC curve 
    tpr_list = np.zeros((10001, num_functions - 1))
    mean_fpr = np.linspace(0, 1, 10001)
    
    for j in range(num_functions-1):
        fpr, tpr, threshold = roc_curve(cat_list[j], prob_list[j])
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        tpr_list[:,j] = tpr     
        
    #Save ROC CURVE
    ROC_df = pd.DataFrame(tpr_list)
    ROC_df['fpr'] = mean_fpr
    ROC_df.columns = categories + ['FPR']
    ROC_df.to_csv(args['model'][:-3] + '_ROC.tsv', sep = '\t') 
    
    print('analysing..', flush = True) 

    #calcualte AUC 
    AUC = statistics.calculate_category_AUC(cat_list, prob_list, categories)
    AUC['all_ovo'] = roc_auc_score(all_cat_list, [i[1:] for i in all_prob_list], multi_class = 'ovo')
    AUC['all_ovr'] = roc_auc_score(all_cat_list, [i[1:] for i in all_prob_list], multi_class = 'ovr')

    #calcualte metrics 
    thresholds = statistics.calculate_thresholds(cat_list, prob_list, categories)
    metrics = statistics.calculate_metrics(all_cat_list, all_prob_list, thresholds, categories, num_functions) 

    #save AUC to a dictionary 
    AUC_file =  args['model'][:-3]  + '_AUC.pkl'
    with open(AUC_file, 'wb') as handle: 
        pickle.dump(AUC, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #save thresholds to a dictionary 
    threshold_file = args['model'][:-3] + '_thresholds.pkl'
    with open(threshold_file, 'wb') as handle: 
        pickle.dump(thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #save the metrics to a dictionary 
    metric_file = args['model'][:-3] + '_metrics.pkl' 
    with open(metric_file, 'wb') as handle: 
        pickle.dump(metrics, handle,protocol=pickle.HIGHEST_PROTOCOL) 

    print('done!', flush  = True)  
    print('\n', flush = True) 

    
if __name__ == "__main__":
    main()
 