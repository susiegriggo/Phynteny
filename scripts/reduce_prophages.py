"""
Script for IBD data prophages 

Get 'high quality' phages with genes from at least four different categories 
"""

import pickle

import click
import pkg_resources

from phynteny_utils import format_data, handle_genbank


@click.command()
@click.option(
    "-i",
    "--input_data",
    help="Text file containing genbank files to build model",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-g",
    "--gene_categories",
    type=int,
    help="Specify the minimum number of categories in each genome",
    default=4,
)
@click.option(
    "--prefix",
    "-p",
    default="data",
    type=str,
    help="Prefix for the output files",
)
def main(input_data, gene_categories, prefix):
    print("STARTING")

    # read in annotations
    phrog_integer = format_data.get_dict(
        pkg_resources.resource_filename(
            "phynteny_utils", "phrog_annotation_info/phrog_integer.pkl"
        )
    )
    phrog_integer = dict(
        zip([str(i) for i in phrog_integer.keys()], phrog_integer.values())
    )
    phrog_integer["No_PHROG"] = 0
    num_functions = len(list(set(phrog_integer.values())))

    # takes a text file where each line is the file path to genbank files of phages to train a model
    print("getting input", flush=True)
    data = handle_genbank.get_data(
        input_data, gene_categories, phrog_integer, False
    )  # dictionary to store all of the training data

    # save the training data dictionary
    print("Done Processing!")
    print("Generating datasets")

    # save the original data
    with open(prefix + "_all_data.pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    # save the keys pf the dictionary to a text file
    filtered_prophages = list(dict(data.keys()))
    with open(prefix + "_all_data_headers.txt", "w") as file:
        for item in filtered_prophages:
            file.write(item + "\n")

    print("DONE")


if __name__ == "__main__":
    main()
