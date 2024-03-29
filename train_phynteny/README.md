Training Phynteny
=============== 

Phynteny has already been trained for you on a dataset containing over 1 million prophages! However, If you feel inclined to train Phynteny yourself you can.

### Generate Training Data
Phynteny is trained using genbank files containing PHROG annotations such as those generated by pharokka. First you will need to generate training data using the `generate_training_data.py` script. This script takes a text file containing the genbank files' paths and includes parameters to specify the maximum number of genes in each prophage, the number of different PHROG categories which should be present in each phage. This outputs four pickled dictionaries, the X components of the testing and training data and the y components of the testing and training data. For example, to generate data from a textfile called `genbank_files.txt` where we would only like to include phages with 120 genes or less with genes from at least four different PHROG categories we would run the command:

```
python generate_training_data.py -i genbank_files.txt -p phynteny_data -max_genes 120 -g 4
```

### Train the model
After this, we can train the model by running the `train_model.py` script. This uses cross-validation to produce several models. The parameters for the LSTM can be adjusted including the numer of hidden layers, memory cells and dropout. For the full list run the command `python scripts/train_model.py --help`. Otherwise we can this command which will generate models with the prefix, `your_trained_phynteny`.
```
python train_model.py -x training_data_X.pkl -y training_data_y.pkl -o your_trained_phynteny
```

**WARNING** Without a GPU training will take a very very long time!

### Compute Confidence
Once you've trained your model you will need to take steps to generate an object specific to your model which can be used to compute the confidence of your predictions. We compute confidence using kernel densities (you can read why this is a good idea [here](https://arxiv.org/abs/2207.06529)). To do this, use the `compute_confidence.py` scripts and parse the testing data.

```
python compute_confidence.py -b model_directory -x testing_data_X.pkl -y testing_data_y.pkl -o confidence_densities
```
