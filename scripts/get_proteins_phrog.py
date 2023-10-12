"""
Get the phrog id, category and protein id of each mmseqs annotated protein
"""

# imports
import pandas as pd
import pickle5 as pickle
import glob

# read in the phrog annotations
annot = pd.read_csv("/home/grig0076/LSTMs/phrog_annot_v4.tsv", sep="\t")
cat_dict = dict(zip([str(i) for i in annot["phrog"]], annot["category"]))
cat_dict[None] = "unknown function"

# integer encoding of each PHROG category
one_letter = {
    "DNA, RNA and nucleotide metabolism": 4,
    "connector": 2,
    "head and packaging": 3,
    "integration and excision": 1,
    "lysis": 5,
    "moron, auxiliary metabolic gene and host takeover": 6,
    "other": 7,
    "tail": 8,
    "transcription regulation": 9,
    "unknown function": 0,
}

# use this dictionary to generate an encoding of each phrog
phrog_encoding = dict(
    zip(
        [str(i) for i in annot["phrog"]], [one_letter.get(c) for c in annot["category"]]
    )
)

# add a None object to this dictionary which is consist with the unknown
phrog_encoding[None] = one_letter.get("unknown function")

# get the directorys containing the genomes
levelone = glob.glob("/home/grig0076/phispy_phrog_pickles/GCA_inc_translation/*")

# get each of the possible phrog categories
all_categories = [
    dict(zip(list(one_letter.values()), list(one_letter.keys()))).get(i)
    for i in range(1, len(one_letter))
]

with open("/home/grig0076/phispy_phrog_pickles/protein_IDs/protein_IDs_phrogs", "w") as f:
    for l1 in levelone:
        leveltwo = glob.glob(l1 + "/*")

        for l2 in leveltwo:
            files = glob.glob(l2 + "/*")

            for file in files:
                with open(file, "rb") as handle:
                    genomes = pickle.load(handle)

                for g in list(genomes.keys()):

                    # get the phrog categories
                    this_genome = genomes.get(g)
                    categories = [
                        phrog_encoding.get(i) for i in this_genome.get("phrogs")
                    ]
                    phrog_ids = this_genome.get("phrogs")
                    protein_ids = this_genome.get("protein_id")

                    for i in range(len(categories)):
                        f.write(protein_ids[i] + '\t' + categories[i] + '\t' + phrog_ids[i])

f.close()