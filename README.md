[![PyPI version](https://badge.fury.io/py/xcolumns.svg)](https://badge.fury.io/py/xcolumns)
[![Documentation Status](https://readthedocs.org/projects/xcolumns/badge/?version=latest)](https://xcolumns.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<p align="center">
  <img src="https://raw.githubusercontent.com/mwydmuch/xCOLUMNs/master/docs/_static/xCOLUMNs_logo.png" width="500px"/>
</p>

# x **Consistent Optimization of Label-wise Utilities in Multi-label classificatioN** s

xCOLUMNs is a small Python library that aims to implement different methods for the optimization of a general family of
metrics that can be defined on multi-label classification matrices.
These include, but are not limited to, label-wise metrics.
The library provides an efficient implementation of the different optimization methods that easily scale to the extreme multi-label classification (XMLC) - problems with a very large number of labels and instances.

All the methods operate on conditional probability estimates of the labels, which are the output of the multi-label classification models.
Based on these estimates, the methods aim to find the optimal prediction for a given test set or to find the optimal population classifier as a plug-in rule on top of the conditional probability estimator.
This makes the library very flexible and allows to use it with any multi-label classification model that provides conditional probability estimates.
The library directly supports numpy arrays, PyTorch tensors, and sparse CSR matrices from scipy as input/output data types.

For more details, please see our short usage guide, the documentation, and/or the papers that describe the methods implemented in the library.


## Quick start

### Installation

The library can be installed using pip:
```sh
pip install xcolumns
```
It should work on all major platforms (Linux, macOS, Windows) and with Python 3.8+.


### Usage

We provide a short usage guide for the library in [short_usage_guide.ipynb](https://github.com/mwydmuch/xCOLUMNs/blob/master/short_usage_guide.ipynb) notebook.
You can also check the documentation for more details.


## Methods implemented in xCOLUMNs

The library implements the following methods:

### Instance-wise weighted prediction

The library implements a set of methods for instance-wise weighted prediction, that include optimal infernece strategies for some metrics, such as:
- Precision at k
- Propensity-scored precision at k
- Macro-averaged recall at k
- Macro-averaged balanced accuracy at k
- and others ...

### Optimization of prediction for a given test set using Block Coordinate Ascent/Descent (BCA/BCD)

The method aims to optimize the prediction for a given metrics and test set using the block coordinate ascent/descent algorithm.

The method was first introduced and described in the paper:
> [Erik Schultheis, Marek Wydmuch, Wojciech Kotłowski, Rohit Babbar, Krzysztof Dembczyński. Generalized test utilities for long-tail performance in extreme multi-label classification. NeurIPS 2023.](https://arxiv.org/abs/2311.05081)

### Finding optimal population classifier using Frank-Wolfe (FW)

The method finds the optimal population classifier for given metric using the Frank-Wolfe optimization algorithm on the provided training set.

The method was first introduced and described in the paper:
> [Erik Schultheis, Wojciech Kotłowski, Marek Wydmuch, Rohit Babbar, Strom Borman, Krzysztof Dembczyński. Consistent algorithms for multi-label classification with macro-at-k metrics. ICLR 2024.](https://arxiv.org/abs/2401.16594)


## Repository structure

The repository is organized as follows:
- `docs/` - Sphinx documentation
- `experiments/` - a code for reproducing experiments from the papers, see the README.md file in the directory for more details
- `xcolumns/` - the library source code
- `tests/` - tests for the library (the coverage is bit limited at the moment, but these test should guarantee that the main components of the library works as expected)


## Development and contributing

The library was created as a part of our research projects.
We are happy to share it with the community and we hope that someone will find it useful.
If you have any questions or suggestions or if you found a bug, please open an issue.
We are also happy to accept contributions in the form of pull requests.
