""" 
Script to train model 

Use to parameter sweep to determine optimal batch size, epochs, dropout, memory cells 
""" 

#imports 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
import tensorflow as tf


def train_model(X_train, y_train, model_file_out, history_file_out, memory_cells, batch_size, epochs, patience, min_delta ): 
    """ 
    Train and save model and training history 
    :param 
    :param 
    """
    
    es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=patience, min_delta=min_delta) #use a set number of epoch when adjusting the memory cells and batch size 
    
    #do we add another layer? 

    #build model 
    model = Sequential() 
    model.add(Bidirectional(CuDNNLSTM(memory_cells, return_sequences=True),input_shape = (max_length, n_features) ))
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(memory_cells, return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(memory_cells, return_sequences = True)))
    model.add(TimeDistributed(Dense(num_functions, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['acc']) 
    print(model.summary(), flush = True)
    
    history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, callbacks = [es], validation_split = 0.1)
    
    #save th
    
    