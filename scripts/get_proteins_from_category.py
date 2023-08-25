""" 
Script which gets the phispy proteins from a specific PHROG 

Used to compare which alphafold protein structures are known 
"""
import glob

# imports
import pandas as pd
import pickle5 as pickle

file_out = "/home/grig0076/phispy_phrog_pickles/protein_IDs/protein_IDs"

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

for c in all_categories:
    print("processing " + c, flush=True)

    category_file_out = file_out + "PHROG_" + c + "_proteinIDs.txt"

    with open(category_file_out, "w") as f:
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

                        # get the proteins of the current category
                        category_idx = [
                            i
                            for i, x in enumerate(categories)
                            if x == one_letter.get(c)
                        ]
                        # translation = [this_genome.get('translation')[c] for c in category_idx]
                        protein_ids = [
                            this_genome.get("protein_id")[c] for c in category_idx
                        ]

                        for p in protein_ids:
                            if p != None:
                                f.write(p)
                                f.write("\n")
    f.close()
