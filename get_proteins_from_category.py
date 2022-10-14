""" 
Script which gets the phispy proteins from a specific VOG category 
""" 


import pandas as pd
import glob
import pickle5 as pickle


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
                 'unknown function' :  0 ,}

#use this dictionary to generate an encoding of each phrog
phrog_encoding = dict(zip([str(i) for i in annot['phrog']], [one_letter.get(c) for c in annot['category']]))

#add a None object to this dictionary which is consist with the unknown
phrog_encoding[None] = one_letter.get('unknown function')

#get the directorys containing the genomes
levelone = glob.glob('/home/grig0076/phispy_phrog_pickles/GCA/*')

#counters to campare the sizes of the training datasets
included = 0
not_included = 0

#dictionary to store the filtered data
data = {}

#make separate text files for the proteins which can be considered as 
with open('/home/grig0076/phispy_phrog_pickles/protein_IDs/PHROG_transcription_category_proteinIDs.txt', 'w') as f:
    
    #loop through each genome
    for l1 in levelone:
        leveltwo = glob.glob(l1+'/*')

        for l2 in leveltwo:

            files = glob.glob(l2+'/*') 

            for file in files: 

                with open(file, 'rb') as handle:
                    genomes = pickle.load(handle)

                for g in list(genomes.keys()):
                    this_genome = genomes.get(g)
                    categories = [phrog_encoding.get(i) for i in this_genome.get('phrogs')]

                    category_idx = [i for i, x in enumerate(categories) if x == one_letter.get('transcription regulation')]
                    protein_id = [this_genome.get('protein_id')[c] for c in category_idx]
                    
                    for protein in protein_id:
                        f.write(protein)
                        f.write('\n')
f.close()
