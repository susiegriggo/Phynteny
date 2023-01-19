""" 
Add mmseqs annotations into phispy annotations 
""" 

#imports 
import pandas as pd 
from pandas.errors  import EmptyDataError
import glob2
from Bio import SeqIO
import re
import gzip 
import pickle
import pathlib
import glob
import numpy as np

def get_mmseqs(phrog_file):
    """
    Check that the dataframe exists
    
    param phrog_filter: location of phrog file to filter
    return: phrog annotations if they exist 
    """
    try:

        phrog_output = pd.read_csv(phrog_file, sep = '\t', compression='gzip', header = None)

    except EmptyDataError:

        phrog_output = pd.DataFrame()
        print('empty mmseqs file: ' + phrog_file)
    
    return phrog_output

def filter_mmseqs(phrog_output, Eval = 1e-5): 
    """ 
    Function to filter the phogs mmseqs output
    If there are two equally good annotations, the annotation with the greatest coverage is taken, then e
    
    param phrog_output: dataframe of phrog annotations 
    param Eval: evalue to filter annotations default (1e-5) 
    return: dictionary of the phrog annotations 
    """
        
    #rename the headers
    phrog_output.columns = ['phrog', 'seq', 'alnScore', 'seqIdentity', 'eVal', 'qStart', 'qEnd', 'qLen', 'tStart', 'tEnd', 'tLen']
    phrog_output['coverage'] = phrog_output['tEnd'] - phrog_output['tStart'] + 1
    phrog_output['phrog'] = [re.split('_', p)[1] for p in phrog_output['phrog']] #convert phrog to a number 
    phrog_output['phrog'] = [re.split('#', p)[0][:-1] for p in phrog_output['phrog']]
    
    #filter to have an e-value lower than e-5 - can change this not to be hardcoded 
    phrog_output = phrog_output[phrog_output['eVal'] < Eval] 

    #filter annotations with multiple hits 
    phrog_output = phrog_output.groupby('seq', as_index = False).coverage.max().merge(phrog_output) #hit with greatest coverage
    phrog_output = phrog_output.groupby('seq', as_index = False).eVal.min().merge(phrog_output) #hit with the best evalue 
    phrog_output = phrog_output.groupby('seq', as_index = False).qLen.min().merge(phrog_output) #hit with the shortest query length 

    return dict(zip(phrog_output['seq'].values, phrog_output['phrog'].values)) 

#read through each genome
directories = glob2.glob('/home/edwa0468/phage/Prophage/phispy/phispy/GCA/' + '/*',  recursive=True)

for d in directories: 
    e = glob2.glob(d + '/**', recursive=True)
    zipped_gdict = [i for i in e if i[-6:] == 'gbk.gz'] 
    
    for file in zipped_gdict: 
        
        print(file, flush = True)  

        #read in the genbank file 
        with gzip.open(file, 'rt') as handle: 
            gb_dict =   SeqIO.to_dict(SeqIO.parse(handle, 'gb'))
            gb_keys = list(gb_dict.keys())

        handle.close() 

        file_parts = re.split('/',file)
        genbank_parts = re.split('_', file_parts[11])
        mmseqs = '/home/edwa0468/phage/Prophage/phispy/phispy_phrogs/GCA/' + file_parts[8] + '/' + file_parts[9] + '/' + file_parts[10] 
        mmseqs_fetch= glob2.glob(mmseqs + '/*')

        if len(mmseqs_fetch) > 0: 

            mmseqs_output  = get_mmseqs(mmseqs_fetch[0]) 

            phrogs = filter_mmseqs(mmseqs_output)

            genbank_name = '/home/grig0076/scratch/phispy_phrogs/GCA/' +  file_parts[8] + '/' + file_parts[9] + '/' + file_parts[10] + '/'  + genbank_parts[0] + '_' + genbank_parts[1] + '_phrogs_' + genbank_parts[3]

            handle = gzip.open(genbank_name, 'wt')

            for key in gb_keys: 

                this_prophage = gb_dict.get(key) 

                #get the cds 
                cds = [i for i in this_prophage.features if i.type == 'CDS']

                #replace the phrogs 
                for c in cds: 

                    if 'protein_id' in c.qualifiers.keys(): 

                        pid = c.qualifiers.get('protein_id')

                        if phrogs.get(pid[0]) == None: 
                            c.qualifiers['phrog'] = 'No_PHROG'
                        else: 
                            c.qualifiers['phrog'] = phrogs.get(pid[0])

            #write to the genbank file 
            SeqIO.write(gb_dict.get(key), handle, 'genbank')

            handle.close() 