"""
Script to parse genbank output
"""

from Bio import SeqIO


def get_genbank(filename):
    """
    Parse genbank file

    param filename: location of genbank file
    return: dictionary of genbank file loci and their features
    """
    with open(filename, 'rt') as handle:
        gb_dict = SeqIO.to_dict(SeqIO.parse(handle, 'gb'))

    return gb_dict


def get_features(genbank):
    """
    Get required features from genbank file

    param genbank: Genbank locus as SeqIO dictionary
    return: Dictionary containing information required for training
    """
    # only consider CDS
    features = [i for i in genbank.features if i.type == 'CDS']

    return {"length": len(genbank.seq),
            "strand": [i.strand for i in features],
            "position": [(int(i.location.start) - 1, int(i.location.end)) for i in features],
            "phrog": [i.qualifiers.get('phrog') for i in features]}


gbk = 'test_data/pharokka.gbk'

gb_dict = get_genbank(gbk)
gb_keys = list(gb_dict.keys())
for key in gb_keys:

    print(get_features(gb_dict.get(key)))
