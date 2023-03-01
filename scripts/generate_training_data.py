import pickle
import re
import click
from phynteny_utils import handle_genbank


def check_positive(ctx, param, value):
    """Type function for click - an integer within some predefined bounds"""
    try:
        value = int(value)
        if value <= 0:
            raise ValueError("Negative input value 0")
        return value
    except ValueError as ex:
        raise click.BadParameter(str(ex))


@click.command()
@click.option(
    "-i",
    "--input",
    help="Text file containing genbank files to build model",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output",
    help="Name of output dictionary containing training data",
    required=True,
    type=click.Path(),
)
@click.option(
    "-max_genes",
    "--maximum_genes",
    type=int,
    help="Specify the maximum number of genes in each genome",
    default=120,
)
@click.option(
    "-gene_cat",
    "--gene_categories",
    type=int,
    help="Specify the minimum number of cateogries in each genome",
    default=4,
)

def main(input, output, maximum_genes, gene_categories):
    print("STARTING")

    # read in annotations
    with open("../phrog_annotation_info/phrog_integer.pkl", "rb") as handle:
        phrog_integer = pickle.load(handle)
        phrog_integer = dict(
            zip([str(i) for i in list(phrog_integer.keys())], phrog_integer.values())
        )

    handle.close()
    phrog_integer["No_PHROG"] = 0

    print("getting input", flush=True)
    # takes a text file where each line is the file path to genbank files of phages to train a model
    print("Extracting...", flush=True)
    print(input, flush=True)

    training_data = {}  # dictionary to store all of the training data

    prophage_counter = 0  # count the number of prophages encountered
    prophage_pass = 0  # number of prophages which pass the filtering steps

    with open(input, "r") as file:

        genbank_files = file.readlines()

        for genbank in genbank_files:

            # convert genbank to a dictionary
            gb_dict = handle_genbank.get_genbank(genbank)
            gb_keys = list(gb_dict.keys())

            for key in gb_keys:

                # update the counter
                prophage_counter += 1

                # extract the relevant features
                phage_dict = handle_genbank.extract_features(gb_dict.get(key))

                # integer encoding of phrog categories
                integer = handle_genbank.phrog_to_integer(
                    phage_dict.get("phrogs"), phrog_integer
                )
                phage_dict["categories"] = integer

                # evaluate the number of categories present in the phage
                categories_present = set(integer)
                if 0 in categories_present:
                    categories_present.remove(0)

                # if above the minimum number of categories are included
                if (
                    len(phage_dict.get("phrogs")) <= maximum_genes
                    and len(categories_present) >= gene_categories
                ):
                    # update the passing candidature
                    prophage_pass += 1

                    # update dictionary with this entry
                    g = re.split(",|\.", re.split("/", genbank.strip())[-1])[0]
                    training_data[g + "_" + key] = phage_dict

    # save the training data dictionary
    print("Done Processing!\n")
    print("Removing duplicate phrog category orders")
    
    # dereplicate the data and shuffle 
    derep_data = handle_genbank.derep_trainingdata(training_data)
    
    # save the original data
    with open(output + "_all_data.pkl", "wb") as handle: 
        pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close() 

    # save the dereplicated data
    with open(output + "_dereplicated.pkl", "wb") as handle:
        pickle.dump(data_derep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

if __name__ == "__main__":
    main()
