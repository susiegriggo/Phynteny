""" 
Module to handle statistics for Phynteny

This module uses code snippets from PHaNNs https://github.com/Adrian-Cantu/PhANNs
""" 

import numpy as np 
import glob 
import pandas as pd 
from sklearn.metrics import classification_report 
from sklearn.metrics import roc_auc_score, roc_curve

def phynteny_score(X_encodings, num_categories, models): 
    """
    calculate the phynteny score. Tests multiple at once as it takes effort to read in the 10 models  
    
    :param encoding: list of encoding matices to generate prediction 
    :param models: list of models which have already been read in 
    :return: per-class phynteny score for the test instance 
    """

    scores_list = [] # list of the score for each prediction 

    for i in range(len(X_encodings)): 

        # get the index we care about for this prediction 
        zero_row = get_masked(X_encodings[i], num_categories) 

        # make prediction for each model 
        yhat = [model.predict(X_encodings[i].reshape(1,120, 16), verbose =0)[0][zero_row] for model in models]

        # get the phynteny score for each category 
        Y_scores = np.array(yhat).sum(axis=0) 
        scores_list.append(Y_scores)

    return np.array(scores_list)

def phynteny_score2():
    """
    To replace the method above
    """

    scores_list = [predict_softmax(X_encodings, num_categories, model)


def known_category(X_encodings, y_encodings, num_categories): 
    """
    Return the category of a masked gene in the test set
    
    :param X_encoding: list of encoded X instances with a single row masked 
    :param y_encoding: list of encoded y instances 
    :return: list of masked categories 
    """
    
    known_category = list() 
    
    for i in range (len(X_encodings)): 
        
        # get the masked index 
        zero_row = get_masked(X_encodings[i], num_categories)
    
        # get the known category of the gene 
        y_index = np.argmax(y_encodings[i][zero_row])
        known_category.append(y_index)
    
    return known_category

def predict_softmax(X_encodings, num_categories, model):
    """
    Predict the function of a masked gene using a single model

    :param encoding: list of encoding matices to generate prediction
    :param model: model object
    :return: softmax prediction tensor
    """

    zero_row = [get_masked(X_encodings[i], num_categories) for i in range(len(X_encodings))]

    X_encodings = np.array(X_encodings)

    yhat = model.predict(X_encodings)

    scores_list = [yhat[i][zero_row[i]] for i in range(len(zero_row))]

    return np.array(scores_list)

def build_roc(scores, known_categories, category_names): 
    """
    Collect values to build the ROC curve 
    
    :param scores: list of values for each category. Can be phynteny scores or softmax scores  
    :param known_categories: Actual label of each sequence 
    :return: dataframe for plotting the ROC curve 
    """
    
    # normalise the scores such that ROC can be computed 
    normed_scores = norm_scores(scores) 
    known_categories = np.array(known_categories)
    
    # for output 
    tpr_list = np.zeros((10001, len(category_names) - 1))
    mean_fpr = np.linspace(0, 1, 10001)
    
    # loop through each category 
    for i in range(1, len(category_names)):
        
        # get items for the category in this iteration  
        include = known_categories == i
        
        # convert the predictions to a series of binary classifications 
        binary_index = [1 for j in known_categories[include]] + [0 for j in known_categories[~include]] 
        binary_scores =  list(normed_scores[include][:,i-1]) + list(normed_scores[~include][:,i-1])
    
        # compute ROC 
        fpr, tpr, thresholds = roc_curve(binary_index, binary_scores)
        
        # store the data 
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        tpr_list[:, i -1] = tpr

    # save the curve for each category to a file 
    ROC_df = pd.DataFrame(tpr_list)
    ROC_df['FPR'] = mean_fpr
    ROC_df.columns = [category_names.get(i) for i in range(len(category_names))][1:] + ['FPR']
    
    return ROC_df 
    
    
def per_category_auc(scores, known_categories, category_names, method = 'ovr'): 
    """
    Calculate the per category under the curve. 
    Calculate the average AUC separately 
    
    :param known_category: known category of each instance
    :param scores: list of either softmax or phynteny scores for each instance 
    :param 
    :return: AUC score for each category  
    """

    # dictionary to store AUC 
    auc_dict = {} 
    
    # normalise the scores such that ROC can be computed 
    normed_scores = norm_scores(scores) 
    known_categories = np.array(known_categories)
    
    # loop through each category 
    for i in range(1, len(category_names)):
        
        # get items for the category in this iteration  
        include = known_categories == i
        
        # convert the predictions to a series of binary classifications 
        binary_index = [1 for j in known_categories[include]] + [0 for j in known_categories[~include]] 
        binary_scores =  list(normed_scores[include][:,i-1]) + list(normed_scores[~include][:,i-1])
    
        # compute AUC  
        auc_dict[category_names.get(i)] = roc_auc_score(binary_index, binary_scores) 
        
    #calculate the average auc 
    auc_dict['average'] = roc_auc_score(known_categories, normed_scores, multi_class = 'ovr') 
    
    return auc_dict

def norm_scores(scores_list): 
    """
    Building a ROC curve in using sklearn requires values to add to 10 and cannot include unknown class.
    Function removes the unknown class and renormalises the output.
    
    :param scores_list: list of softmax or phynteny scores 
    """ 
    
    return scores_list[:,1:]/scores_list[:,1:].sum(axis=1)[:, np.newaxis]  
    
def get_masked(encoding, num_categories): 
    """ 
    Get which indexes are masked in the data. Important  pre-masked testing data/ 
    
    :param encoding: encoded matrix 
    :num_categories: number of gene functional categories in the encoding  
    :return: list of masked indexes 
    """ 
    
    return np.where(np.all(encoding[:,:num_categories] == 0, axis=1))[0][0]
        
    
def class_scores(tt,scores,is_real,prot_class,df):
    """
    Function for scoring quality of predictions and geting metrics 
    Modified from PhANNs https://github.com/Adrian-Cantu/PhANNs/blob/master/model_training/08_graph.py
    
    :param tt: threshold cutoff to apply 
    :param is_real: 
    :param prot_class: cateogory to predict from 
    :param df: dataframe to append to 
    """
    
    is_predicted=[x>=tt-0.05 for x in scores]
    TP=sum(np.logical_and(is_real,is_predicted))
    FN=sum(np.logical_and(is_real,np.logical_not(is_predicted)))
    TN=sum(np.logical_and(np.logical_not(is_real),np.logical_not(is_predicted)))
    FP=sum(np.logical_and(np.logical_not(is_real),is_predicted))

    if not (TP+TN+FP+FN):
        return df
    
    num_pred=TP+FP
    
    if not num_pred:
        precision=0
        
    else:
        precision=TP/num_pred
        
    num_rec=(TP+FN)
    
    if not num_rec:
        recall=0
        
    else:
        recall=TP/num_rec

    fscore=(2*TP)/(2*TP+FP+FN)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    data_row=[prot_class,precision,recall,fscore,accuracy,tt]
    df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    
    return df

def threshold_metrics(scores, known_categories, category_names): 
    """
    Calculate various metrics at different Phynteny scores 
    Modified from PhANNs https://github.com/Adrian-Cantu/PhANNs/blob/master/model_training/08_graph.py
    
    :param scores: phytneny scores for each category 
    :param known_categories: Actual label of each sequence 
    :param category_names: dictionary of category labels 
    """
    
    d = {'class':[],'precision': [], 'recall': [],'f1-score':[],'accuracy':[],'threshold':[]}

    score_range=np.arange(0,10.1,0.1)
    df_test_score = pd.DataFrame(data=d)

    scores_index = np.array([np.argmax(i) for i in scores])
    known_categories = np.array(known_categories)

    # loop through each category and take the predictions made to that class (regardless whether successful)
    for num in range(1,len(category_names)): 

        test_set_p = scores[scores_index == num, num]
        test_set_t = known_categories[scores_index==num]==num

        for tt in score_range:

            df_test_score=class_scores(tt,np.around(test_set_p[test_set_p>=tt-0.05],decimals=1)
                                                       ,test_set_t[test_set_p>=tt-0.05]
                                                       ,num,df_test_score)

    df_test_score['class'] = [int(i) for i in df_test_score['class']]
    df_test_score['category'] = [category_names.get(i) for i in df_test_score['class'] ] 

    return df_test_score 
    