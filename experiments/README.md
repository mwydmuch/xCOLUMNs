# Experiments

This directory contains the code for reproducing experiments from the following papers:

- Erik Schultheis, Marek Wydmuch, Wojciech Kotłowski, Rohit Babbar, Krzysztof Dembczgitński. _Generalized test utilities for long-tail performance in extreme multi-label classification_. NeurIPS 2023.


## Requirements

The scripts require to have xCOLUMNs package installed:
```sh
pip install xcolumns
```
as well as few additional packages listed in `requirements.txt` file. They can be installed using:
```sh
pip install -r requirements.txt
```

## Strucuture

- `datasets/` - the scripts expect to find datasets in this directory
- `predictions/` - the scripts expect to find probability estimes from different models (e.g. LightXML) in this directory
- `results/` - the scripts save the results of different inference methods in this directory using json format
- `notebooks/` - contains jupyter notebooks used for additional analysis and visualization of the results
- `run_bc_experiment.py` - script for running the experiments using Block Coordinate Ascent (BCA) inference method
- `summarize_results` - script that creates tables data from the results of the experiments


## Datasets

TODO


## LightXML predictions for Block Coordinate experiments

TODO
