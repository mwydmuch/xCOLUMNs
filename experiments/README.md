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
- `run_neurips_2023_bca_experiment.py` - main script for running experiments using Block Coordinate Ascent (BCA) inference method from the NeurIPS 2023 paper,
- `run_all_neurips_2023_bca_experiments.sh` - script for running all the experiments from the NeurIPS 2023 paper,
- `run_iclr_2024_fw_experiment.py` - script for running the experiments using Frank Wolfe algorithm (FW) inference method from the ICLR 2024 paper,
- `run_all_iclr_2024_fw_experiments.sh` - script for running all the experiments from the ICLR 2024 paper,
- `run_icml_2024_omma_experiment.py` - main script for the experiments using OMMA online inference method from the ICML 2024 paper,
- `run_all_iclr_2024_omma_experiments.sh` - script for running all the experiments from the ICLR 2024 paper,
- `run_thesis_2024_experiment.py` - main script for running experiments from Marek Wydmuch's PhD thesis,
- `run_all_thesis_2024_experiments_org.sh` - script for running experiments on original datasets from Marek Wydmuch's PhD thesis,
- `run_all_thesis_2024_experiments_syn.sh` - script for running synthetic experiments from Marek Wydmuch's PhD thesis,
- `run_all_thesis_2024_experiments_syn_sample.sh` - script for reevaluating all the synthetic experiments with different true labels resapled,
- `run_all_thesis_2024_experiments_top_labels.sh` - script for running all the experiments comparing all vs top labels for the 2024 thesis,
- `summarize_results.py` - script that creates tables data from the results of the experiments.


## Reproducing the Block Coordinate Ascent (BCA) experiments from the NeurIPS 2023 paper

To replicate the experiments from the NeurIPS 2023 paper titled "Generalized test utilities for long-tail performance in extreme multi-label classification," you need to download the following datasets and predictions:
- Download standard (BOW) versions of the datasets (Eurlex-4K, AmazonCat-13K, Wiki10-31K, Wikipedia-500K, Amazon-670K) from [here](https://drive.google.com/drive/folders/1fI8ZqiEPqdLeDhUi7wVk8-jhSpvBtNxh?usp=drive_link) or [XMLC Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), unpack, and place them in the `datasets/` directory.
- Download LightXML predictions from [here](https://drive.google.com/drive/folders/1bcOUYCjcePnlZHU4yW8TigWAiuGI-36T?usp=drive_link) and place them in the `predictions/lighxml` directory.

Now you can use `run_neurips_2023_bca_experiment.py` script to run the specific experiment. Of just run `run_all_neurips_2023_bca_experiments.sh` to run all the experiments. The results of every separate experiment (single run of a single method of specified dataset) will be saved in the `results/bca_neurips` directory as a json files.


## Reproducing the Frank Wolfe (FW) experiments from the ICLR 2024 paper

To replicate the experiments from the ICLR 2023 paper titled "Consistent algorithms for multi-label classification with macro-at-k metrics," you need to download the following datasets:
- TODO: Upload and provide the link to the datasets.

Now you can use `run_iclr_2024_fw_experiment.py` script to run the specific experiment. Of just run `run_all_iclr_2024_fw_experiments.sh` to run all the experiments. The results of every separate experiment (single run of a single method of specified dataset) will be saved in the `results/fw_iclr` directory as a json files.
