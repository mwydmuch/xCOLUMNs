# Experiments

This directory contains the code for reproducing experiments from the following papers:

> [Erik Schultheis, Marek Wydmuch, Wojciech Kotłowski, Rohit Babbar, Krzysztof Dembczyński. Generalized test utilities for long-tail performance in extreme multi-label classification. NeurIPS 2023.](https://arxiv.org/abs/2311.05081)

and

> [Erik Schultheis, Wojciech Kotłowski, Marek Wydmuch, Rohit Babbar, Strom Borman, Krzysztof Dembczyński. Consistent algorithms for multi-label classification with macro-at-k metrics. ICLR 2024.](https://arxiv.org/abs/2401.16594)

and some additional experiments.


## Requirements

The scripts require to have xCOLUMNs package installed:
```sh
pip install xcolumns
```
as well as few additional packages listed in `requirements.txt` file. They can be installed using:
```sh
pip install -r requirements.txt
```

## Directory and files

- `datasets/` - the scripts expect to find datasets in this directory,
- `predictions/` - the scripts expect to find probability estimes from different models (e.g. LightXML) in this directory,
- `results/` - the scripts save the results of different inference methods in this directory using json format,
- `notebooks/` - contains jupyter notebooks used for additional analysis and visualization of the results,
- `run_neurips_bc_experiment.py` - script for running the experiments using Block Coordinate Ascent (BCA) inference method from the NeurIPS 2023 paper,
- `run_all_neurips_bc_experiments.sh` - script for running all the experiments from the NeurIPS 2023 paper,
- `summarize_results.py` - script that creates tables data from the results of the experiments.


## Reproducing the Block Coordinate Ascent (BCA) experiments from the NeurIPS 2023 paper

To replicate the experiments from the NeurIPS 2023 paper titled "Generalized test utilities for long-tail performance in extreme multi-label classification," you need to download the following datasets and predictions:
- Download standard (BOW) versions of the datasets (Eurlex-4K, AmazonCat-13K, Wiki10-31K, Wikipedia-500K, Amazon-670K) from [here](https://drive.google.com/drive/folders/1fI8ZqiEPqdLeDhUi7wVk8-jhSpvBtNxh?usp=drive_link) or [XMLC Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), unpack, and place them in the `datasets/` directory.
- Download LightXML predictions from [here](https://drive.google.com/drive/folders/1bcOUYCjcePnlZHU4yW8TigWAiuGI-36T?usp=drive_link) and place them in the `predictions/lighxml` directory.

Now you can use `run_neurips_bc_experiment.py` script to run the specific experiment. Of just run `run_all_neurips_bc_experiments.sh` to run all the experiments. The results of every separate experiment (single run of a single method of specified dataset) will be saved in the `results/` directory as a json files.
