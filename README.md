[![Edwards Lab](https://img.shields.io/badge/Bioinformatics-EdwardsLab-03A9F4)](https://edwards.sdsu.edu/research)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/548652990.svg)](https://zenodo.org/badge/latestdoi/548652990)
![GitHub language count](https://img.shields.io/github/languages/count/susiegriggo/Phynteny) 
[![CI](https://github.com/susiegriggo/Phynteny/actions/workflows/testing.yml/badge.svg)](https://github.com/susiegriggo/Phynteny/actions/workflows/testing.yml)


<p align="center">
  <img src="https://raw.githubusercontent.com/susiegriggo/Phynteny/no_unknowns/phynteny_logo.png" width="600" title="phynteny logo" alt="phynteny logo">
</p>

Phynteny: Synteny-based annotation of bacteriophage genes 
=============== 

Approximately 65% of all bacteriophage (phage) genes cannot be attributed a known biological function. Phynteny uses a long-short term memory model trained on phage synteny (the conserved gene order across phages) to assign hypothetical phage proteins to a [PHROG](https://phrogs.lmge.uca.fr/) category. 

Phynteny is still a work in progress and the LSTM model has not yet been optimised. Use with caution! 

**NOTE** This version of Phynteny will only annotate phages with 120 genes or less due to the architecture of the LSTM. We aim to adjust this in future versions. 


## Dependencies
Phynteny installation requires Python 3.8 or above. You will need the following python dependencies to run Phynteny and its related support scripts. The latest tested versions of the dependencies are: 
* [python](https://www.python.org/) - version 3.10.0 
* [sklearn](https://scikit-learn.org/stable/) - version 1.2.2 
* [biopython](https://biopython.org/) - version 1.81
* [numpy](https://numpy.org/) - version 1.21.0 (Windows, Linux, Apple Intel), version 1.24.0 (Apple M1/M2)
* [tensorflow](https://www.tensorflow.org/) - version 2.9.0 (Windows, Linux, Apple Intel), tensorflow-macos version 2.11 (Apple M1/M2)
* [pandas](https://pandas.pydata.org/) - version 2.0.2
* [loguru](https://loguru.readthedocs.io/en/stable/) - version 0.7.0
* [click](https://click.palletsprojects.com/en/8.1.x/) - version 8.1.3 <br> 

We recommend GPU support if you are training Phynteny. This requires CUDA and cuDNN:
* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) - version 11.2 
* [cuDNN](https://developer.nvidia.com/cudnn) - version 8.1.1 

## Installation 

Currently Phynteny can be installed from this repository
```
git clone https://github.com/susiegriggo/Phynteny.git --branch main --depth 1 
cd Phynteny 
pip install . 
```

### Install Models 
Once you've installed Phynteny you'll need to download the pre-trained models
```
install_models.py 
```
If you would like to specify a particular location to download the models run
```
install_models.py -o <path/to/database_dir>
```

If for some reason this does not work. you can download the pre-trained models from [Zenodo](https://zenodo.org/record/8198288/files/phynteny_models_v0.1.11.tar.gz) and untar in a location of your choice. 


## Usage 

Phynteny takes a genbank file containing PHROG annotations as input. If you phage is not yet in this format, [pharokka](https://github.com/gbouras13/pharokka) can take your phage (in fasta format) to a genbank file with PHROG annotations.  Phynteny will then return a genbank files and a table containing the details of the predictions made using phynteny. Each prediction is accompanied by a 'phynteny score' which ranges between 1-10 and a recalibrated confidence score. 

**Reccomended**  
```
phynteny test_data/test_phage.gbk  -o test_phynteny
```

**Custom** 

If you wish to specify your own LSTM model, run: 

```
phynteny test_phage.gbk -o test_phage_phynteny -m your_models -t confidence_dict.pkl 
```
Details of how to train the phynteny models and generate confidence estimates is detailed below. 

## Train Phynteny 

Phynteny has already been trained for you on a dataset containing over 1 million prophages! If you feel inclined to generate your own Phynteny model using your own dataset, instructions and training scripts are provided [here](https://github.com/susiegriggo/Phynteny/tree/no_unknowns/train_phynteny).

## Performance 

Coming soon: Notebooks demonstrating the performance of the model 

## Bugs and Suggestions 

If you break Phynteny or would like to make any suggestions please open an issue or email me at susie.grigson@flinders.edu.au 

## Wow! How can I cite this incredible piece of work? 

The Phynteny manuscript is currently in preparation. In the meantime, please cite Phynteny as: 
```
Grigson, S. R.,  Mallawaarachchi, V., Roach, M. R., Papudeshi, B., Bouras, G., Decewicz, P., Dinsdale, E. A. & Edwards, R. A. (2023). Phynteny: Synteny-based annotation of phage genomes. DOI: 10.5281/zenodo.8128917
```

If you use pharokka to annotate your phage before using Phynteny please cite it as well: 
```
Bouras, G., Nepal, R., Houtak, G., Psaltis, A. J., Wormald, P. J., & Vreugde, S. (2023). Pharokka: a fast scalable bacteriophage annotation tool. Bioinformatics, 39(1), btac776.
```

