"""
Convert PHROG mmseqs annotations to dictionaries to train LSTM 

Combines prophage in genbank file and mmseqs PHROGs annotation into a dictionary which contains details of each prophage

Working progress. Needs to be able to take Phispy with mmseqs or just genbank files

Make a separate script which inserts the phrog annotations into the genbank files
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


def check_df(phrog_file):
    """
    Check that the dataframe exists
    """
    try:

        phrog_output = pd.read_csv(phrog_file, sep = '\t', compression='gzip', header = None)

    except EmptyDataError:

        phrog_output = pd.DataFrame()
        print('empty mmseqs file: ' + phrog_file)
    
    return phrog_output

def get_PHROGs(phrog_output): 
    """ 
    Function to filter the phogs mmseqs output 
    """

    #rename the headers
    phrog_output.columns = ['phrog', 'seq', 'alnScore', 'seqIdentity', 'eVal', 'qStart', 'qEnd', 'qLen', 'tStart', 'tEnd', 'tLen']
    phrog_output['coverage'] = phrog_output['tEnd'] - phrog_output['tStart'] + 1
    phrog_output['phrog'] = [re.split('_', p)[1] for p in phrog_output['phrog']]
    phrog_output['phrog'] = [re.split('#', p)[0][:-1] for p in phrog_output['phrog']]

    #filter annotations with multiple hits 
    phrog_output = phrog_output.groupby('seq', as_index = False).coverage.max().merge(phrog_output) #hit with greatest coverage
    phrog_output = phrog_output.groupby('seq', as_index = False).eVal.min().merge(phrog_output) #hit with the best evalue 
    phrog_output = phrog_output.groupby('seq', as_index = False).qLen.min().merge(phrog_output) #hit with the shortest query length 
    
    return dict(zip(phrog_output['seq'], phrog_output['phrog'])) 

#read through each genome
base = 'glob.glob('/home/edwa0468/phage/Prophage/phispy/phispy/GCA' #TODO - hardcoded - have as something which can be parsed in
all_files = glob2.glob(base + '/**',  recursive=True)

#only consider dictionaries or zipped dictionaries
dict = [i for i in all_files if i[-4:] == '.gbk']
zipped_dict = [i for i in all_files if i[-6:] == 'gbk.gz']



#match this to mmseqs output


""""""
level_one = glob.glob('/home/edwa0468/phage/Prophage/phispy/phispy/GCA/*')
for l1 in level_one: 
    
    level_two = glob.glob(l1+ '/*')
    
    
    for l2 in level_two:
        
        l2_genome_dict = {} 
        level_three = glob.glob(l2+'/*') 
        
        for l3 in level_three: 

            #get path to genbank file 
            genbank = [i for i in glob.glob(l3 + '/*') if 'VOGS_phage.gbk.gz' in i]

            if len(genbank) > 0: 
                
                genbank = genbank[0] 
            
                #get path to phrogs file 
                file_parts = re.split('/',l3)
                this_file = file_parts[7] + '_' + file_parts[8]+file_parts[9] + file_parts[10]
                phrogs = '/home/edwa0468/phage/Prophage/phispy/phispy_phrogs/GCA' + '/' + file_parts[8] + '/' + file_parts[9] + '/' + file_parts[10] 
            
                mmseqs = glob.glob(phrogs + '/*') 
 
                if len(mmseqs) > 0: 
                    
                    mmseqs = glob.glob(phrogs + '/*')[0]

                    #check dataframe
                    phrog_output = check_df(mmseqs) 

                    if len(phrog_output) > 0: 
                                        
                        #get the phrog dictionary 
                        phrog_dict = get_PHROGs(phrog_output) 

                        #get the details for the genbank file 
                        with gzip.open(genbank, 'rt') as handle: 
                            gb_dict =   SeqIO.to_dict(SeqIO.parse(handle, 'gb'))
                            gb_keys = list(gb_dict.keys())

                            # loop through each contig
                            for key in gb_keys:

                                #get record for the current contigs
                                this_contig = gb_dict.get(key)
                                contig_length = len(this_contig.seq)
                                this_CDS = [i for i in this_contig.features if i.type == 'CDS']

                                position = [(int(this_CDS[i].location.start), int(this_CDS[i].location.end)) for i in range(len(this_CDS))] 
                                sense = [re.split(']',str(this_CDS[i].location))[1][1] for i in range(len(this_CDS))] 
                                protein_id = [this_CDS[i].qualifiers.get('protein_id') for i in range(len(this_CDS))] 
                                protein_id = [p[0] if p is not None else None for p in protein_id]
                                phrogs = [phrog_dict.get(p) for p in protein_id]

                                #formulate a dictionary with this information 
                                contig_dict = {'length': contig_length, 'phrogs' : phrogs, "protein_id": protein_id, "sense": sense, "position": position}

                                #update dictionary with this one 
                                l2_genome_dict[this_file + '_' + key] = contig_dict
         
        #save the informatiogn to a pickle file at the dictionary level 
        dict_name = 'GCA' + '_' + file_parts[8] + file_parts[9] +  file_parts[10] + '.pkl'  
        
        #create location 
        dict_location = '../phispy_phrog_pickles/GCA/' + file_parts[8] + '/' + file_parts[9]
        pathlib.Path(dict_location).mkdir(parents=True, exist_ok=True) 
        dict_name = dict_location + '/' + dict_name
        
        with open(dict_name, 'wb') as handle:
            pickle.dump(l2_genome_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
""" 
Save this as one greate big dictionary