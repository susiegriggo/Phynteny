#!/usr/bin/env python3

"""
Generate training data for the model
"""

# imports
import pickle
import argparse
import re
from phynteny_utils import handle_genbank

def check_positive(arg):
    """Type function for argparse - a float within some predefined bounds"""

    value = int(arg)
    if value <= 0:
        raise argparse.ArgumentTypeError("Negative input value 0")
    return value

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate training data for retraining the Phynteny model"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Text file containing genbank files to build model",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Name of output dictionary containing training data",
        required=True,
    )
    parser.add_argument(
        "-max_genes",
        "--maximum_genes",
        type=check_positive,
        help="Specify the maximum number of genes in each genome",
        required=False,
        default=120,
    )
    parser.add_argument(
        "-gene_cat",
        "--gene_categories",
        type=check_positive,
        help="Specify the minimum number of cateogries in each genome",
        required=False,
        default=4,
    )
    return vars(parser.parse_args())

def main():

    print('STARTING') 

    # get arguments
    args = parse_args()

    # read in annotations
    with open("../phrog_annotation_info/phrog_integer.pkl", "rb") as handle:
        phrog_integer = pickle.load(handle)
        phrog_integer = dict(
            zip([str(i) for i in list(phrog_integer.keys())], phrog_integer.values())
        )
    handle.close()
    phrog_integer['No_PHROG'] = 0 

    print('getting input', flush = True)  
    # takes a text file where each line is the file path to genbank files of phages to train a model
    print("Extracting...", flush=True)
    print(args["input"], flush=True)

    training_data = {}  # dictionary to store all of the training data
    
    prophage_counter = 0 # count the number of prophages encountered 
    prophage_pass = -0 # number of prophages which pass the filtering steps 


    with open(args["input"], "r") as file:

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
                    len(phage_dict.get("phrogs")) <= args["maximum_genes"]
                    and len(categories_present) >= args["gene_categories"]
                ):

                    # update the passing candidature
                    prophage_pass += 1 

                    # update dictionary with this entry
                    g = re.split(",|\.", re.split("/", genbank.strip())[-1])[0]
                    training_data[g + "_" + key] = phage_dict


    # save the training data dictionary
    print("Done Processing!\n")
    print("Removing duplicate phrog category orders")

    derep_data = handle_genbank.derep_trainingdata(training_data) #TODO go back and check this dereplication procedure - how can I make sure it is random
    data_derep_shuffle = handle_genbank.shuffle_dict(derep_data)

    # save the non-dereplicated data - this might be what ends up being used 
    with open(args["output"] + "_all_data.pkl", "wb") as handle: 
        pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close() 

    with open(args["output"] + "_dereplicated.pkl", "wb") as handle:
        pickle.dump(data_derep_shuffle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    print("\nTraining data save to " + str(args["output"] + "_all_data.pkl"))

    print("Complete!")
    print(
        str(prophage_counter)
        + " phages parsed. "
        + str(prophage_pass)
        + " phages used. " 
        + str(len(list(data_derep_shuffle.keys())))
        + " phages left after dereplication." 
    )

if __name__ == "__main__":
    main()
