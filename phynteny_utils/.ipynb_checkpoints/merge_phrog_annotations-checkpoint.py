""" 
Add mmseqs annotations into phispy annotations 
""" 

#imports 
import glob2
from Bio import SeqIO, bgzf 
import re
import gzip
import mmseqs_output

#read through each genome
directories = glob2.glob('/home/edwa0468/phage/Prophage/phispy/phispy/GCA/' + '/*',  recursive=True)

for d in directories:

    e = glob2.glob(d + '/**', recursive=True)
    zipped_gdict = [i for i in e if i[-6:] == 'gbk.gz'] 
    
    for file in zipped_gdict: 

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

            annotations  = mmseqs_output.get_mmseqs(mmseqs_fetch[0])
            
            if len(annotations) > 0: 
                
                phrogs = mmseqs_output.filter_mmseqs(annotations)
                genbank_name = '/home/grig0076/scratch/phispy_phrogs/GCA/' +  file_parts[8] + '/' + file_parts[9] + '/' + file_parts[10] + '/'  + genbank_parts[0] + '_' + genbank_parts[1] + '_phrogs_' + genbank_parts[3]
                
                #handle = gzip.open(genbank_name, 'wt')

                #loop through each prophage
                for key in gb_keys: 

                    #get a prophage
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
                print(genbank_name, flush = True)  
                with gzip.open(genbank_name, 'wt') as handle: 
                    SeqIO.write(gb_dict.get(key), handle, 'genbank')
                handle.close() 