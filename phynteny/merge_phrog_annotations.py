""" 
Add mmseqs annotations into phispy annotations 
"""

# imports
import glob2
from Bio import SeqIO, bgzf
import re
import gzip
import handle_genbank

# read through each genome
directories = glob2.glob('/home/edwa0468/phage/Prophage/phispy/phispy/GCA/' + '/*', recursive=True)

for d in directories:

    e = glob2.glob(d + '/**', recursive=True)
    zipped_gdict = [i for i in e if i[-6:] == 'gbk.gz']

    for file in zipped_gdict:

        # convert genbank to a dictionary
        gb_dict = handle_genbank.get_genbank(file)
        gb_keys = list(gb_dict.keys())

        file_parts = re.split('/', file)
        genbank_parts = re.split('_', file_parts[11])
        mmseqs = '/home/edwa0468/phage/Prophage/phispy/phispy_phrogs/GCA/' + file_parts[8] + '/' + file_parts[9] + '/' + \
                 file_parts[10]
        mmseqs_fetch = glob2.glob(mmseqs + '/*')

        if len(mmseqs_fetch) > 0:

            if os.path.getsize(mmseqs_fetch[0]):
                annotations = handle_genbank.get_mmseqs(mmseqs_fetch[0])

                if len(annotations) > 0:

                    phrogs = handle_genbank.filter_mmseqs(annotations)
                    genbank_name = '/home/grig0076/scratch/phispy_phrogs/GCA/' + file_parts[8] + '/' + file_parts[
                        9] + '/' + file_parts[10] + '/' + genbank_parts[0] + '_' + genbank_parts[1] + '_phrogs_' + \
                                   genbank_parts[3]

                    # write to the genbank file
                    print(genbank_name, flush=True)
                    with gzip.open(genbank_name, 'wt') as handle:

                        # loop through each prophage
                        for key in gb_keys:

                            # get a prophage
                            this_prophage = gb_dict.get(key)

                            # get the cds
                            cds = [i for i in this_prophage.features if i.type == 'CDS']

                            # loop through each protein
                            for c in cds:

                                if 'protein_id' in c.qualifiers.keys():

                                    pid = c.qualifiers.get('protein_id')

                                    if phrogs.get(pid[0]) == None:
                                        c.qualifiers['phrog'] = 'No_PHROG'
                                    else:
                                        c.qualifiers['phrog'] = phrogs.get(pid[0])

                            # write genbank file
                            SeqIO.write(gb_dict.get(key), handle, 'genbank')

                    handle.close()
