'''
Script to split up the dereplicated dictionary 
'''

# imports 
import pickle5
from phynteny_utils import format_data

# dereplicate dictionary
derep = pickle5.load(open('/home/grig0076/scratch/phynteny_data/many_to_one_data/phynteny_data_dereplicated.pkl', "rb"))
derep_keys = list(derep.keys())

# need to encode all of the data in the dereplicated dictionary
derep_X, derep_y = format_data.generate_dataset(derep, 10, 120)
X_dict = dict(zip(derep_keys, derep_X))
y_dict = dict(zip(derep_keys, derep_y))

# save the dereplicated X and Y data to a dictionary
# Open the file in binary write mode ('wb')
with open('/home/grig0076//scratch/phynteny_data/many_to_one_data/phynteny_data_dereplicated_X.pkl', 'wb') as file:
    # Use the pickle.dump() method to save the dictionary to the file
    pickle5.dump(X_dict, file)

# Open the file in binary write mode ('wb')
with open('/home/grig0076/scratch/phynteny_data/many_to_one_data/phynteny_data_dereplicated_y.pkl', 'wb') as file:
    # Use the pickle.dump() method to save the dictionary to the file
    pickle5.dump(y_dict, file)
