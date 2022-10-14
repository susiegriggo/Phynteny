""" 
Functions to prepare data for training with the LSTM viral gene organisation model
""" 

#imports 
import numpy as np 
import random 
from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_strand(strand): 
    """ 
    One hot encode sense
    
    :param strand: sense encoded as a vector of 1s and 2s 
    :return: one hot encoding as two separate numpy arrays
    """ 
    
    return np.array([1 if i==1 else 0 for i in strand]), np.array([1 if i==2 else 0 for i in strand])

def derep_trainingdata(training_data, phrog_encoding): 
    """ 
    Ensure there is only one copy of each phrog order and sense order 
    
    :param training_data: dictionary containing training data 
    :param phrog_encoding: dictionary which converts phrogs to category integer encoding 
    :return: dereplicated training dictionary 
    """
    
    #get the training keys and encodings 
    training_keys = list(training_data.keys()) 
    training_encodings = [[phrog_encoding.get(i) for i in training_data.get(key).get('phrogs')] for key in training_keys] 
    
    #write a function to remove duplicates in the training data 
    training_str = [''.join([str(j) for j in i]) for i in training_encodings]
    training_sense = [''.join(training_data.get(p).get('sense')) for p in training_keys]
    training_hash = [training_sense[i] + training_str[i] for i in range(len(training_keys))]

    #get the dereplicated keys 
    dedup_keys = list(dict(zip(training_hash, training_keys)).values())

    return dict(zip(dedup_keys, [training_data.get(d) for d in dedup_keys]))

def format_data(training_data, phrog_encoding): 
    """ 
    Intial function to generate training data.
    Currently only includes genomes which start or end with an integrase. This is hard coded and will likely need changing. 
    
    :param training_data: dictionary which contains details for each genome 
    :param phrog_encoding: dictionary which converts phrogs to cateogory integer encoding 
    :return: training encodings one-hot encoding each genome 
    :return: list of features 
    """
    
    training_encodings = []
    sense_encodings = []
    start_encodings = []
    length_encodings = []
    intergenic_encodings = [] 

    training_keys = list(training_data.keys()) 

    for key in training_keys: 

        encoding = [phrog_encoding.get(i) for i in training_data.get(key).get('phrogs')]
        length = np.array([i[1] - i[0] for i in training_data.get(key).get('position')])

        #if the integrase is at the end then reverse the sequence 
        if encoding[-1] == 1: 

            #flip the order and gene lengths 
            encoding = encoding[::-1]
            length = length[::-1] 

            #encode the strand  
            sense = np.array([1 if i == '+' else 2 for i in training_data.get(key).get('sense')])
            sense = sense[::-1]

            #get the start positions 
            start = np.array([training_data.get(key).get('length') - i[1] + 1for i in training_data.get(key).get('position')])
            start = start[::-1]

            #intergenic distances 
            intergenic = [training_data.get(key).get('position')[i+1][0] - training_data.get(key).get('position')[i][1]  for i in range(len(training_data.get(key).get('position')[::-1]) -1 )]
            intergenic.insert(0,0) 

        else: 
            #encode the strand 
            sense = np.array([2 if i == '+' else 1 for i in training_data.get(key).get('sense')])

            #start position of each gene 
            start = np.array([i[0] - training_data.get(key).get('position')[0][0] + 1 for i in training_data.get(key).get('position')])

            #intergenic distances 
            intergenic = [training_data.get(key).get('position')[i+1][0] -  training_data.get(key).get('position')[i][1]  for i in range(len(training_data.get(key).get('position'))-1)]  
            intergenic.insert(0, 0)
            

        #update the features 
        training_encodings.append(encoding) 
        sense_encodings.append(sense) 
        start_encodings.append(start) 
        intergenic_encodings.append(intergenic) 
        length_encodings.append(length)

    #scale the lengths such that the maximum length is 1 
    max_length = np.max([np.max(l) for l in length_encodings])
    length_encodings = [l/max_length for l in length_encodings]

    #divide intergenic distance by the absolute maximum 
    max_intergenic = np.max([np.max(np.abs(i)) for i in intergenic_encodings]) 
    intergenic_encodings = [i/max_intergenic for i in intergenic_encodings]

    #scale the start positions according to the length of the genome 
    start_encodings = [s/np.max(s) for s in start_encodings] #simply divide starts by the length of the sequence 

    #split the sense into two separate features as it is categorical data 
    sense_encodings = [encode_strand(s) for s in sense_encodings]
    strand1s = [s[0] for s in sense_encodings]
    strand2s = [s[1] for s in sense_encodings] 

    #return a set of features to train the LSTM 
    features = [strand1s, strand2s, length_encodings, start_encodings, intergenic_encodings] 

    return training_encodings, features 

def one_hot_encode(sequence, n_features):
    """ 
    One hot encode PHROG categories as data is cateogrical. 
    
    :param sequence: numerical sequence of PHROG cateogories 
    :param n_features: total number of features in the model
    :return: numpy array containing one hot encoding 
    """ 
    
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_features)]
        vector[value] = 1
        encoding.append(vector)
        
    return np.array(encoding)
    
    
def encode_feature(encoding, feature, column): 
    """ 
    Add a feature to sequence feature matrix 
    
    :param encoding: matrix including features for some seuqnece
    :param feature: feature to append to matrix 
    :param column: column to add feature 
    :return: feature matrix including the new feature  
    """ 

    encoding = encoding.astype('float64')
    encoding[:len(feature), column] = feature

    return encoding 


def one_hot_decode(encoded_seq):
    """ 
    Return one-hot encoding of PHROG category to its original numeral value 
    
    :param encoded_seq: one_hot encoding of the sequence 
    :return: integer encoding of the PHROG cateogries present in a sequence 
    """ 
    return [np.argmax(vector) for vector in encoded_seq]


def generate_example(sequence, features, num_functions, n_features, max_length, func): 
    """ 
    Convert a sequence of PHROG functions and associated features to a supervised learning problem 
    
    :param sequence: integer encoded list of PHROG categories in a sequence
    :param features: list of features to include in problem 
    :param num_functions: number of possible PHROG categories  
    :param max_length: maximum length of a sequence 
    :return: training or test example separated as X and y matrices 
    """

    seq_len = len(sequence) 
    padded_sequence = pad_sequences([sequence], padding = 'post', maxlen = max_length)[0]
    y = np.array(one_hot_encode(padded_sequence, num_functions))

    if func in padded_sequence:
     
        occurence = [i for i, x in enumerate(padded_sequence) if x == func]
        idx = random.choice(occurence) 
        func = padded_sequence[idx] 
    
    
        """
        #introduce a single masked value into the sequence 
        idx = random.randint(1, seq_len -1) #don't include ends


        while padded_sequence[idx] == 0: 
            idx = random.randint(1, seq_len -1)
        """ 
    
        X =  np.array(one_hot_encode(padded_sequence, n_features ))

        for f in range(len(features)): 
            X = encode_feature(X, features[f], num_functions + f) 

        #replace the function encoding for the masked sequence 
        X[idx, 0:num_functions] = np.zeros(num_functions) 
        
        #reshape the matrices 
        X = X.reshape((1, max_length, n_features))
        y = y.reshape((1, max_length, num_functions))
        
    else: 
        
        X = None 
        y = None 
        func = None 
    
    return X, y, func
    
def generate_prediction(sequence, features, num_functions, n_features, max_length): 
    """ 
    Prepare data to predict the function of all hypothetical genes in a sequence 
    
    :param sequence: inteer enoded list of PHROG categories in a sequence 
    :param features: list of features to include 
    :param num_functions: number of possible PHROG categories 
    :param max_length: maximum length of a sequence 
    :return: encoded matrix which can be parsed to the model
    """ 
    
    #construct features 
    seq_len = len(sequence) 
    padded_sequence = pad_sequences([sequence], padding = 'post', maxlen = max_length)[0]
    X =  np.array(one_hot_encode(padded_sequence, n_features ))
    for f in range(len(features)): 
        X = encode_feature(X, features[f], num_functions + f)

    #mask each unknown in the sequence  
    unknown_idx = [i for i, x in enumerate(sequence) if x == 0]
    for unk in unknown_idx: 
        X[unk, 0:num_functions] = np.zeros(num_functions) 
 
    return X.reshape((1, max_length, n_features)) 

    
def generate_dataset(sequences, all_features, num_functions, n_features, max_length): 
    """" 
    Generate a dataset to train LSTM model 
    
    :param sequences: set of sequences encoded as integers for each PHROG
    :param all_features:  set of features to include in the encodings 
    :param num_functions: number of possible PHROG categories 
    :param n_features: total number of features 
    :param max_length: maximum length of a sequence 
    :return: Dataset of training or test data reprsented as X and y matrices 
    """
    
    #features is a list of list objects 
    X = [] 
    y = [] 
    masked_idx = [] 

    #generate the function to mask in this genome 
    func = random.randint(1, num_functions-1)
        
    for i in range(len(sequences)): 
        
        this_X, this_y, idx = generate_example(sequences[i], all_features[i], num_functions, n_features, max_length, func) 
        

        if this_X is not None: 
           
            X.append(this_X) 
            y.append(this_y)
            masked_idx.append(idx) 
            
            #generate a new function for the next training example 
            func = random.randint(1, num_functions-1)

    X = np.array(X).reshape(len(masked_idx),max_length,n_features)
    y = np.array(y).reshape(len(masked_idx), max_length, num_functions)
    
    return X, y, masked_idx 