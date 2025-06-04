<p align="center">
  <img src="https://raw.githubusercontent.com/susiegriggo/Phynteny/no_unknowns/phynteny_logo.png" width="600" title="phynteny logo" alt="phynteny logo">
</p>

Phynteny: Synteny-based annotation of bacteriophage genes 
[![Edwards Lab](https://img.shields.io/badge/Bioinformatics-EdwardsLab-03A9F4)](https://edwards.sdsu.edu/research)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/548652990.svg)](https://zenodo.org/badge/latestdoi/548652990)
![GitHub language count](https://img.shields.io/github/languages/count/susiegriggo/Phynteny) 
[![CI](https://github.com/susiegriggo/Phynteny/actions/workflows/testing.yml/badge.svg)](https://github.com/susiegriggo/Phynteny/actions/workflows/testing.yml)
[![PyPI version](https://badge.fury.io/py/phynteny.svg)](https://badge.fury.io/py/phynteny)
[![Downloads](https://static.pepy.tech/badge/phynteny)](https://pepy.tech/project/phynteny)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/phynteny/badges/version.svg)](https://anaconda.org/bioconda/phynteny)
![Conda](https://img.shields.io/conda/dn/bioconda/phynteny)
=============== 

***READ THIS*** New version of Phynteny, [Phynteny Transformer](https://github.com/susiegriggo/Phynteny_transformer) is now available. This should provide more accurate results. 


Approximately 65% of all bacteriophage (phage) genes cannot be attributed a known biological function. Phynteny uses a long-short term memory model trained on phage synteny (the conserved gene order across phages) to assign hypothetical phage proteins to a [PHROG](https://phrogs.lmge.uca.fr/) category. 

Phynteny is still a work in progress and the LSTM model has not yet been optimised. Use with caution! 

**NOTE:** This version of Phynteny will only annotate phages with 120 genes or less due to the architecture of the LSTM. We aim to adjust this in future versions. 

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
### Option 1: Installing Phynteny using conda (recommended)
You can install Phynteny from bioconda at [https://anaconda.org/bioconda/phynteny](https://anaconda.org/bioconda/phynteny). Make sure you have [`conda`](https://docs.conda.io/en/latest/) installed. 
```bash
# create conda environment and install phynteny 
conda create -n phynteny -c bioconda phynteny
 
# activate environment
conda activate phynteny

# install phynteny
conda install -c bioconda phynteny
```

**NOTE:** bioconda installations of Phynteny do not have GPU support. This is fine for most uses but not does not enable training of phynteny models. 

Now you can go to [Install Models](#install-models) to install pre-trained phynteny models. 

### Option 2: Installing Phynteny using pip
You can install Phynteny from PyPI at [https://pypi.org/project/phynteny/](https://pypi.org/project/phynteny/). Make sure you have [`pip`](https://pip.pypa.io/en/stable/) and [`mamba`](https://mamba.readthedocs.io/en/latest/index.html) installed.

```
pip install phynteny
```

**NOTE:** pip installation is recommended for training Phynteny models 

Now you can go to [Install Models](#install-models) to install pre-trained phynteny models. 

### Option 3: Installing Phynteny from source 
If all else fails you can install Phynteny from this repo. 

```
git clone https://github.com/susiegriggo/Phynteny.git --branch main --depth 1 
cd Phynteny 
pip install . 
```

Now you can go to [Install Models](#install-models) to install pre-trained phynteny models. 

### Install Models 
Once you've installed Phynteny you'll need to download the pre-trained models
```
install_models 
```
If you would like to specify a particular location to download the models run
```
install_models -o <path/to/database_dir>
```

If for some reason this does not work. you can download the pre-trained models from [Zenodo](https://zenodo.org/record/8198288/files/phynteny_models_v0.1.11.tar.gz) and untar in a location of your choice. 

## Usage 

Phynteny takes a genbank file containing PHROG annotations as input. If your phage is not yet in this format, [pharokka](https://github.com/gbouras13/pharokka) can take your phage (in fasta format) to a genbank file with PHROG annotations.  Phynteny will then return a genbank files and a table containing the details of the predictions made using phynteny. Each prediction is accompanied by a 'phynteny score' which ranges between 1-10 and a recalibrated confidence score. 

**Reccomended**  
```
phynteny tests/data/test_phage.gbk  -o test_phynteny
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

If you found Phynteny useful and would like to get even better annotations for your phages check out [phold](https://github.com/gbouras13/phold)!

