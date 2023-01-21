"""
Generate training data for the model
""" 

#imports
import pickle
import argparse
import handle_genbank

def check_positive(arg):
    """ Type function for argparse - a float within some predefined bounds """

    value = int(arg)
    if value <= 0:
        raise argparse.ArgumentTypeError("Negative input value 0" )
    return value

#argparser
parser = argparse.ArgumentParser(description='Generate training data for retraining the Phynteny model')
parser.add_argument('-i', '--input', help = 'Text file containing genbank files to build model', required=True)
parser.add_argument('-o', '--output', help='Name of output dictionary containing training data', required=True)
#parser.add_argument('-max_genes', '--maximum_genes', type=check_positive, help='Specify the maximum number of genes in each genome', required=False, default=120)
#parser.add_argument('-gene_cat', '--gene_categories', type=check_positive, help='Specify the minimum number of cateogries in each genome', required=False, default=4)
#arser.add_argument('-c', '--chunks', type=check_positive, help='Number of chunks to divide data', required=False, default=0)
args = vars(parser.parse_args())

training_data = {} #dictionary to store all of the training data

#takes a textfile where each line is the file path to genbank files of phages to train a model
with open(args['input']) as file:

    for genbank in file:

        #convert genbank to a dictionary
        gb_dict = handle_genbank.get_genbank()
        gb_keys = list(gb_dict.keys())

        for key in gb_keys:

                #extract the relevant features
                phage_dict = handle_genbank.extract_features(gb_dict.get(key)

                # update dictionary with this one
                g = re.split(',|\.', re.split('/', genbank.strip())[-1])[0]
                training_data[g + '_' + key] = phage_dict

#save the training data dictionary
with open(args['output'] + '_all_data.pkl', 'wb') as handle:
    pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done! \nTraining data saved to ' + str(args['output']))


