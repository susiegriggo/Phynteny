""" 
Generate separate test and training dataset for training the model 
Requires GPU to run 
""" 


#imports 
import format_data 
import argparse 


parser = argparse.ArgumentParser(description='Generate test and train datasets without duplicate genomes')
parser.add_argument('-phrogs','--phrogs_annotations', help='Location of relevant phrogs training data file', required=True)
parser.add_argument('-g', '--genome_dictionary_location', help = 'Location where processed genome dictionaries are saved', required = True)
parser.add_argument('-rep', '--replicates', help = 'Number of replicate testing and training datasets', required = True)
parser.add_argument('-train', '--training', help = 'Number of genomes in training set', required = True) 
parser.add_argument('-test', '--testing', help = 'Number of genomes in testing set', required = True) 
    
args = vars(parser.parse_args())

test_portion = 0.9
train_portion = 1 - test_portion 

#read in the phrog annotations
annot = pd.read_csv(args.phrogs, sep = '\t')
cat_dict = dict(zip([str(i) for i in annot['phrog']], annot['category']))
cat_dict[None] = 'unknown function'

#integer encoding of each PHROG category
one_letter = {'DNA, RNA and nucleotide metabolism' : 4,
 'connector' : 2,
 'head and packaging' : 3,
 'integration and excision': 1,
 'lysis' : 5,
 'moron, auxiliary metabolic gene and host takeover' : 6,
 'other' : 7,
 'tail' : 8,
 'transcription regulation' : 9,
 'unknown function' :  0 ,}

#use this dictionary to generate an encoding of each phrog
phrog_encoding = dict(zip([str(i) for i in annot['phrog']], [one_letter.get(c) for c in annot['category']]))

#add a None object to this dictionary which is consist with the unknown
phrog_encoding[None] = one_letter.get('unknown function')

#obtain the data 
data = data.generate_data(args.g)

#split into training and test datasets 
keys = list(data.keys()) 

#randomly reshuffle these keys 
for rep in range(0,args.rep): 

    random.shuffle(drep_keys) 

    #select training and test data 
    train_num = int(len(derep_keys)* train_portion) 
    training_keys = derep_keys[:train_num] 
    test_keys = derep_keys[train_num:] 

    #get the train and test dictionaries 
    train_data = dict(zip(training_keys, [data.get(key) for key in training_keys])) 
    test_data = dict(zip(test_keys, [data.get(key) for key in test_keys]))

    #pickle these two dictionaries 
    with open (args.train, 'wb') as handle: 
        pickle.dump(train_data, handle, protocol = pickle.HIGHEST_PROTOCOL) 

    with open(args.test, 'wb') as handle: 
        pickle.dump(test_data, handle, protocol = pickle.HIGHEST_PROTOCOL) 