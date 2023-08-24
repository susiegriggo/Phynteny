"""
Remove duplicate protein ids 
"""

import glob

# imports
import pandas as pd

files = glob.glob("/home/grig0076/phispy_phrog_pickles/protein_IDs/uniq/*")
files = [f for f in files if "PHROG" in f]

with open(
    "/home/grig0076/phispy_phrog_pickles/protein_IDs/uniq/duplicate_checksums.txt", "w"
) as f:
    checksums = []
    for i in range(len(files)):
        checksums.append(set(pd.read_csv(files[i], header=None)[0].to_list()))

        if len(checksums) > 0:
            for j in range(i):
                a = checksums[i].intersection(checksums[j])

                if len(a) > 0:
                    for c in list(a):
                        f.write(c)
                        f.write("\n")

f.close()
