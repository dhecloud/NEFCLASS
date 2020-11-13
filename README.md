# Implementation of NEFCLASS in python


This repository implements the neuro-fuzzy [NEFCLASS](https://www.researchgate.net/publication/221000724_NEFCLASS_-_a_neuro-fuzzy_approach_for_the_classification_of_data) model in python

## Requisites
First install anaconda onto your system.  
```
conda create -n nefclass python==3.6.10
conda install pandas scikit-learn
```

Bulk of the code is done in pure numpy. `pandas` is used for loading csv files, and `scikit-learn` is used for creating kfolds cross validation sets.

## Data folder
Git clone this repository and put the [wine dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) or the [iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris) in `data/`. If you want to extend to your own dataset, you can add the data loading function to `data_loading.py` and then add the appropriate function to `main.py`

The data folder should look like this
```
data/Iris.csv
data/wine/winequality-red.csv
data/wine/winequality-white.csv
data/wine/winequality.names
```


## Commands 
```
usage: main.py [-h] [--dataset DATASET] [--sigma SIGMA]
               [--num_epoch NUM_EPOCH] [--num_sets NUM_SETS] [--kmax KMAX]
               [--rule_learning RULE_LEARNING] [--cv] [--kfold KFOLD] [-v]
               [--mf MF]

NEFCLASS

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset to load
  --sigma SIGMA         learning rate
  --num_epoch NUM_EPOCH
                        number of epoch for fuzzy set learning
  --num_sets NUM_SETS   number of fuzzy sets
  --kmax KMAX           maximum number of rules
  --rule_learning RULE_LEARNING
                        rule learning method to use. Default is the original
                        implementation. Use any other strings as input to
                        perform best per class.
  --cv                  do 10 fold cross validation?
  --kfold KFOLD         number of k fold
  -v                    verbosity
  --mf MF               membership function to use. Default: tri. Options:
                        gaussian, semicircle
```

### Example commands

To train on iris dataset on default arguments with verbose printing  
`python main.py -v`

To perform cross validation on iris dataset   
`python main.py --cv`

To train on wine dataset, with best per class rule learning, and learning rate of 0.1;   
`python main.py -v --dataset wine --rule_learning bestperclass --sigma 0.1`


## Acknowledgements
This project was done as a part of the fulfilment of a module at Nanyang Technological University