""" 
Module to handle statistics for Phynteny

This module uses code snippets from PHaNNs https://github.com/Adrian-Cantu/PhANNs
""" 

import numpy as np 
import glob 
from sklearn.metrics import classification_report 

def phynteny_score(encoding, idx, models): 
    """
    calculate the phynteny score. Tests multiple at once as it takes effort to read in the 10 models  
    
    :param encoding: list of encoding matices to generate prediction 
    :param idx: which index in the phage to make the predictions for 
    :param models: path to where the replicate models are located 
    :return: per-class phynteny score for the test instance 
    """
    
    x = 1
    
    
def get_masked(encoding, num_categories): 
    """ 
    Get which indexes are masked in the data. Important for pre-masked testing data/ 
    
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
    try:
        specificity=TN/(TN+FP)
        
    except ZeroDivisionError:
        specificity=0
        
    false_positive_rate=FP/(FP+TN)
    fscore=(2*TP)/(2*TP+FP+FN)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    data_row=[prot_class,precision,recall,fscore,specificity,false_positive_rate,accuracy,tt]
    df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    
    return df

def make_predictions(): 
    """
    Make some predictions using the Phynteny model
    """
    
    
    
    