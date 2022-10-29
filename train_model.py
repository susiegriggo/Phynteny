""" 
Script to train model 

Use to parameter sweep to determine optimal batch size, epochs, dropout, memory cells 
""" 

#imports 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
import tensorflow as tf

import pickle 
import argparse 
import random 
import format_data
import numpy as np 

def train_model(X_train, y_train, X_test, y_test, max_length, n_features, num_functions, model_file_out, history_file_out, memory_cells, batch_size, epochs, dropout, recurrent, lr, early, patience, min_delta ): 
    """ 
    Train and save model and training history 
    
    :param X_train: Supervised learning problem features 
    :param y_train: Supervised learning problem labels
    :param model_file_out: File name of saved model
    :param history_file_out: File name of history dictionary
    :param memory cells: Number of memory cells to use for each LSTM layer
    :param batch_size: Batch size for training
    :param epochs: Number of epochs to use for training
    :param patience: Stopping condition - how many epochs without loss increasing to stop training 
    :param min_delta: Stopping condition loss value
    """
    
    model = Sequential() 
    model.add(Bidirectional(LSTM(memory_cells, return_sequences=True, dropout = dropout, kernel_regularizer=L1L2(0, 0)),input_shape = (max_length, n_features )))
    model.add(Bidirectional(LSTM(memory_cells, return_sequences = True, dropout = dropout, kernel_regularizer=L1L2(0, 0))))
    model.add(Bidirectional(LSTM(memory_cells, return_sequences = True, dropout = dropout, kernel_regularizer=L1L2(0, 0))))

    model.add(TimeDistributed(Dense(num_functions, activation='softmax')))
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy']) 
    print(model.summary(), flush = True)
    
    
    mc = ModelCheckpoint(model_file_out[:-3] + 'best_val_accuracy.h5',
                        monitor='val_loss', mode='min', save_best_only=True, verbose=1) #model with the best validation loss therefore minimise 
    mc2 = ModelCheckpoint(model_file_out[:-3] + 'best_val_loss.h5',
                        monitor='val_accuracy', mode='max', save_best_only=True, verbose=1) #model with the best validation set accuracy therefore maximise 
    
    if early == True: 
        es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=patience, min_delta=min_delta) #use a set number of epoch when adjusting the memory cells and batch size 
        history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, callbacks = [es, mc, mc2], validation_data = (X_test, y_test))
    else: 
        history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.1)
        
    #save the model 
    model.save(model_file_out) 
    
    #save the history dictionary as a pickle 
    with open(history_file_out, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def evaluate_predictions(test_encodings, test_features, num_functions, n_features, max_length, model):
    """
    Evaluate the accuracy of a trained LSTM. Does not use unbiased category predictions 
    
    :param test_encodings: encoded gene orders of a testing set of genomes 
    :param test_features: encoded features for a set of testing genomes 
    :param num_functions: number of possible functional categories 
    :param n_features: number of features in addition to the functional categories 
    :param max_length: maximum number of genes in a seqeunces 
    :param model: trained model 
    :return % score of the accuracy of the model 
    """ 
    
    correct = 0 
    
    for i in range(len(test_encodings)): 

        idx = random.randint(1, len(test_encodings[i]) -1) #don't include ends

        #make sure that the mask is not an uknown category 
        while test_encodings[i][idx] == 0: 
            idx = random.randint(1, len(test_encodings[i]) -1)

        X, y = format_data.generate_example(test_encodings[i], test_features[i], num_functions, n_features, max_length, idx) 
        yhat = model.predict(X) 

        #the pre decoded sequences from softmax act as probabilities 
        if [np.argmax(i) for i in yhat[0]] == [np.argmax(i) for i in y[0]]: 
            correct += 1 
        
    return (correct/len(test_encodings))*100.0
    
def evaluate_predictions_unbiased_categories(test_encodings, test_features, num_functions, n_features, max_length, model): 
    """ 
    Evaluate the accuracy of a trained LSTM. Uses an equal number of each category for predictions. 
    
    :param test_encodings: encoded gene orders of a testing set of genomes 
    :param test_features: encoded features for a set of testing genomes 
    :param num_functions: number of possible functional categories 
    :param n_features: number of features in addition to the functional categories 
    :param max_length: maximum number of genes in a seqeunces 
    :param model: trained model 
    :return % score of the accuracy of the model 
    """
    
    #evaluate the model for an unbiased category 
    correct = 0 
    predictions = 0 
    func = random.randint(1, num_functions-1)
    
    for i in range(len(test_encodings)): 

        if func in test_encodings[i]:

            occurence = [i for i, x in enumerate(test_encodings[i]) if x == func]
            idx = random.choice(occurence) 

            X, y = format_data.generate_example(training_encodings[i], features[i], num_functions, n_features, max_length, idx) 
            yhat = model.predict(X) 

            #the pre decoded sequences from softmax act as probabilities 
            if [np.argmax(i) for i in yhat[0]] == [np.argmax(i) for i in y[0]]: 
                correct += 1 

            #prepare for the next prediction 
            func = random.randint(1, num_functions-1)
            predictions += 1 

    print('Accuracy: %f' % ((correct/predictions)*100.0))

