"""
Generate training data for the model
""" 

#imports
from Bio import SeqIO
import re
import gzip 
import pickle
import argparse

#argparser
parser = argparse.ArgumentParser(description='Generate training data for retraining the Phynteny model')
parser.add_argument('-i', '--input', help = 'Text file containing genbank files to build model', required=True)
parser.add_argument('-o', '--output', help='Name of output dictionary containing training data', required=True)
args = vars(parser.parse_args())

training_data = {} #dictionary to store all of the training data

#takes a textfile where each line is the file path to genbank files of phages to train a model
with open(args['input']) as file:

    for genbank in file:
        
        if genbank.strip()[-3:] == '.gz': 
            try: 
                with gzip.open(genbank.strip(), 'rt') as handle:
                    gb_dict = SeqIO.to_dict(SeqIO.parse(handle, 'gb'))      
            except ValueError:     
                print('ERROR: ' + genbank.strip() +' is not a genbank file!') 
                raise
        
        else:       
            try: 
                with open(genbank.strip(), 'rt') as handle:
                    gb_dict = SeqIO.to_dict(SeqIO.parse(handle, 'gb'))
            except ValueError: 
                print('ERROR: ' + genbank.strip() +' is not a genbank file!')
                raise

        
        # loop through each phage
        gb_keys = list(gb_dict.keys())
        for key in gb_keys:

                this_phage = gb_dict.get(key)
                phage_length = len(this_phage.seq)
                this_CDS = [i for i in this_phage.features if i.type == 'CDS'] #coding sequences

                position = [(int(this_CDS[i].location.start), int(this_CDS[i].location.end)) for i in range(len(this_CDS))]
                sense = [re.split(']', str(this_CDS[i].location))[1][1] for i in range(len(this_CDS))]
                protein_id = [this_CDS[i].qualifiers.get('protein_id') for i in range(len(this_CDS))]
                protein_id = [p[0] if p is not None else None for p in protein_id]
                phrogs = [this_CDS[i].qualifiers.get('phrog') for i in range(len(this_CDS))]
                phrogs = ['No_PHROG' if i is None else i[0] for i in phrogs]

                # formulate a dictionary with this information
                phage_dict = {'length': phage_length, 'phrogs': phrogs, "protein_id": protein_id, "sense": sense,
                               "position": position}

                # update dictionary with this one
                g = re.split(',|\.', re.split('/', genbank.strip())[-1])[0]
                training_data[g + '_' + key] = phage_dict

#save the training data dictionary
with open(args['output'], 'wb') as handle:
    pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
#perform filtering of the training dataset 
    
print('Done! \nTraining data saved to ' + str(args['output'])) 