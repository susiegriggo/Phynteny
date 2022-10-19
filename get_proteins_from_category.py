""" 
Script which gets the phispy proteins from a specific PHROG 
""" 
#imports 
import pandas as pd
import glob
import pickle5 as pickle
import re
from Bio import SeqIO 
import gzip
from crc64iso.crc64iso import crc64

base = '/home/edwa0468/phage/Prophage/phispy/fasta_protein/GCA' #where fasta files are located 
file_out = '/home/grig0076/phispy_phrog_pickles/protein_IDs/PHROG_connector_proteinIDs.txt'
category = 'connector' 

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

with open(file_out, 'w') as f:

    for l1 in levelone:
        leveltwo = glob.glob(l1+'/*')

        for l2 in leveltwo:

            files = glob.glob(l2+'/*') 

            for file in files: 

                with open(file, 'rb') as handle:
                    genomes = pickle.load(handle)

                for g in list(genomes.keys()):

                    #get the corresponding protein fasta file 
                    fasta_file = glob.glob(base + '/' + re.split('_', g)[1][0:3] + '/' + re.split('_', g)[1][3:6] + '/' + re.split('_', g)[1][6:] + '/*')[0] 
                    proteins =  SeqIO.to_dict(SeqIO.parse(gzip.open(fasta_file, "rt"), "fasta"))

                    #get the phrog categories
                    this_genome = genomes.get(g)
                    categories = [phrog_encoding.get(i) for i in this_genome.get('phrogs')]

                    #get the proteins 
                    category_idx = [i for i, x in enumerate(categories) if x == one_letter.get(category)]
                    protein_id = [this_genome.get('protein_id')[c] for c in category_idx]

                    #calcuate the checksum for each protein 

                    for p in protein_id: 
                        seq = proteins.get(p)
                        f.write(p) 
                        f.write('\t') 
                        f.write(crc64(str(seq.seq)))
                        f.write('\n')
f.close() 