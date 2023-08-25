"""
Script to merge mmseqs annotations with a genbank file
"""

import gzip

import click
from Bio import SeqIO

# imports
from phynteny_utils import handle_genbank


@click.command()
@click.option("--genbank", "-g", type=click.Path(), help="genbank file")
@click.option("--out", "-o", type=click.Path(), help="output genbank file")
def main(genbank, out):
    emax = 1e-10
    print("reading genbank", flush=True)
    gb_dict = handle_genbank.get_genbank(genbank)
    gb_keys = gb_dict.keys()

    print("getting mmseqs", flush=True)
    print("looping", flush=True)

    with gzip.open(out, "wt") as handle:
        # loop through the genome
        for k in gb_keys:
            this_phage = gb_dict.get(k)
            features = [i for i in this_phage.features]

            # loop through features in the genome
            for f in features:
                # change type to gene
                f.type = "CDS"

                if "phrog_id" in f.qualifiers.keys():
                    if float(f.qualifiers["evalue"][0]) < emax:
                        f.qualifiers["phrog"] = f.qualifiers["phrog_id"][0][6:]
                    else:
                        f.qualifiers["phrog"] = "No_PHROG"

                else:
                    f.qualifiers["phrog"] = "No_PHROG"

            # add an evalue filtering step here as well
            # write genbank file
            SeqIO.write(gb_dict.get(k), handle, "genbank")

    handle.close()


if __name__ == "__main__":
    main()
