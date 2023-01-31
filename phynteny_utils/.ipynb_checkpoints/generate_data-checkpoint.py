""" 
Generate training data from a set of processed genomes
Each instance included must include at least four different PHROG categories. 
Resulting instances are dereplicated such that each prophage included has a unique permutation of gene orders 
Dictionary is saved into 11 different chunks of the same size for subsequent 10-fold cross validation

Incorporate this with the other script
"""

#imports
import pandas as pd 
import pickle
import glob
import format_data 

base = '/home/grig0076/phispy_phrog_pickles/cross_validated/prophage_phrog_data_derep_fourormore_lessthan121.pkl'
k = 11 #TODO don't have this hardcoded

#read in the phrog annotations 
annot = pd.read_csv('/home/grig0076/LSTMs/phrog_annot_v4.tsv', sep = '\t') 
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
 'unknown function' :  10,}

#use this dictionary to generate an encoding of each phrog
phrog_encoding = dict(zip([str(i) for i in annot['phrog']], [one_letter.get(c) for c in annot['category']]))

#add a None object to this dictionary which is consist with the unknown 
phrog_encoding[None] = one_letter.get('unknown function') 

levelone = glob.glob('/home/grig0076/phispy_phrog_pickles/GCA/*')

#dictionary to store training data 
data = {} 

for l1 in levelone: 
    leveltwo = glob.glob(l1+'/*') 
    
    for l2 in leveltwo: 
        
        file = glob.glob(l2+'/*')[0]
        with open(file, 'rb') as handle: 
            prophages = pickle.load(handle) 
            
            
        for g in list(prophages.keys()): 
            this_prophage = prophages.get(g)
            
            categories = [phrog_encoding.get(i) for i in this_prophage.get('phrogs')]
            
            categories_present = set(categories) 
            if 10 in categories_present:
                categories_present.remove(10)
                
            #consider only prophages which have at least four of the different annotations 
            if len(categories_present) >= 4 and len(categories) <= 120: 
                
                    data[g] = this_prophage  

print('Data includes: ' + str(len(data)) + ' prophages') 

#remove duplicates 
data_derep= format_data.derep_trainingdata(data, phrog_encoding) 
#shuffle data 
data_derep_shuffle = format_data.shuffle_dict(data_derep)

print('Data includes: ' + str(len(data_derep_shuffle)) + ' prophages after dereplication') 

with open(base, 'wb') as handle:
            pickle.dump(data_derep_shuffle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#generate chunks of data for cross validation 
k = 11 
n = int(len(data_derep_shuffle)/11)
suffix = [i for i in range(k-1)]
suffix.append('test') 

for i in range(k): 
    
    fold = dict(list(data_derep_shuffle.items())[i*n: (i+1)*n])
    
    filename = base[:-4] + '_' + str(suffix[i]) + '_chunk.pkl'
    filehandler = open(filename,"wb")
    pickle.dump(fold, filehandler)

