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
from collections import ChainMap


def train_kfold(base, phrog_encoding, k, num_functions, n_features, max_length, file_out,  memory_cells, batch_size, epochs, dropout, recurrent, lr, early, patience, min_delta, features = 'all'): 
    """ 
    Separate training data into 
    
    :param k: Number of folds to train over 
    :param base: base string of location of the data chunks 
    :param phrog_encoding: dictionary mapping proteins to phrog categories 
    :param num functions: number of possible functions inlcuded in data 
    :param n_features: number of possible features included in data
    :param max_length: maximum number of genes included in a single prophage 
    :param file_out: destination to save model outputs 
    :param memory cells: Number of memory cells to use for each LSTM layer
    :param batch_size: Batch size for training
    :param epochs: Number of epochs to use for training
    :param patience: Stopping condition - how many epochs without loss increasing to stop training 
    :param min_delta: Stopping condition loss value
    """ 
    
    #loop through the training chunks 
    kk = np.array(range(k))
    for i in kk: 
    
        chunks = [base + str(f) + '_chunk.pkl' for f in kk[kk!=i]]
        print('reading chunks', flush = True) 
        training_chunks = [pickle.load(open(f, 'rb')) for f in chunks]
        
        print('merging chunks', flush = True)
        train_data = ChainMap(*training_chunks)
            
        #generate the training data
        print('generating data', flush = True) 
        train_encodings, train_features = format_data.format_data(train_data, phrog_encoding) 
        X_train, y_train, masked_idx = format_data.generate_dataset(train_encodings, train_features, num_functions, n_features, max_length, features)
        
        #generate the test data 
        test_data = pickle.load(open(base + str(i) + '_chunk.pkl', 'rb'))
            
        test_encodings, test_features = format_data.format_data(test_data, phrog_encoding) 
        X_test, y_test, masked_idx = format_data.generate_dataset(test_encodings, test_features, num_functions, n_features, max_length, features) 

        #file names of the data 
        model_file_out = file_out + str(i) + '_' + features + '_chunk_trained_LSTM.h5' 
        history_file_out = file_out + str(i) + '_' + features + '_chunk_history.pkl' 
    
        print('Training for chunk: ' + str(i), flush = True) 
        
        #train the model 
        train_model(X_train, y_train, X_test, y_test, max_length, n_features, num_functions, model_file_out, history_file_out, memory_cells, batch_size, epochs, dropout, recurrent, lr, early, patience, min_delta)
        
        del X_train 
        del y_train 
        del train_encodings 
        del train_features 
        
        del X_test
        del y_test 
        del test_encodings 
        del test_features 
    
    
def train_model(X_train, y_train, X_test, y_test, max_length, n_features, num_functions, model_file_out, history_file_out, memory_cells, batch_size, epochs, dropout, recurrent, lr, early, patience, min_delta): 
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
    

    es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=patience, min_delta=min_delta) #use a set number of epoch when adjusting the memory cells and batch size 
    history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, callbacks = [es, mc, mc2], validation_data = (X_test, y_test))
        
    #save the model 
    model.save(model_file_out) 
    
    #save the history dictionary as a pickle 
    with open(history_file_out, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        