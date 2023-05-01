"""
Script to merge mmseqs annotations with a genbank file
"""
# imports
from phynteny_utils import handle_genbank
from Bio import SeqIO
import gzip

genbank = ''
mmseqs = ''
out_name = ''

gb_dict = handle_genbank.get_genbank(genbank)
gb_keys = gb_dict.keys()

annotations = handle_genbank.get_mmseqs(mmseqs)
phrogs = handle_genbank.filter_mmseqs(annotations)

with gzip.open(out_name, 'wt') as handle:

    for k in gb_keys:

        this_phage = gb_keys.get(k)
        cds = [i for i in this_phage.features if i.type == 'CDS']
        for c in cds:

            if 'protein_id' in c.qualifiers.keys():

                pid = c.qualifiers.get('protein_id')

                if phrogs.get(pid[0]) == None:
                    c.qualifiers['phrog'] = 'No_PHROG'
                else:
                    c.qualifiers['phrog'] = phrogs.get(pid[0])

        # write genbank file
        SeqIO.write(gb_dict.get(k), handle, 'genbank')

handle.close()

