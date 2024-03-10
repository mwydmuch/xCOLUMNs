[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/xcolumns.svg)](https://badge.fury.io/py/xcolumns)

<p align="center">
  <img src="https://raw.githubusercontent.com/mwydmuch/xCOLUMNs/master/xCOLUMNs_logo.png" width="500px"/>
</p>

# x **Consistent Optimization of Label-wise Utilities in Multi-label classificatioN** s

xCOLUMNs is a small Python library that aims to implement different methods for the optimization of a general family of
metrics that can be defined on multi-label classification matrices.
These include, but are not limited to, label-wise metrics.
The library provides an efficient implementation of the different optimization methods that easily scale to the extreme multi-label classification (XMLC) - problems with a very large number of labels and instances.

All the methods operate on conditional probability estimates of the labels, which are the output of the multi-label classification models.
Based on these estimates, the methods aim to find the optimal prediction for a given test set or to find the optimal population classifier as a plug-in rule on top of the conditional probability estimator.
This makes the library very flexible and allows to use it with any multi-label classification model that provides conditional probability estimates.
The library directly supports numpy arrays, PyTorch tensors (work in progress), and sparse CSR matrices from scipy as input/output data types.

For more details, please see our short usage guide, the documentation, and/or the papers that describe the methods implemented in the library.


## Quick start

### Installation

The library can be installed using pip:
```sh
pip install xcolumns
```
It should work on all major platforms (Linux, macOS, Windows) and with Python 3.8+.


### Usage

We provide a short usage guide for the library in [short_usage_guide.ipynb]() notebook.
You can also check the documentation for more details.


## Methods, usage, and how to cite

The library implements the following methods:

### Instance-wise weighted prediction

The library implements a set of methods for instance-wise weighted prediction, that include optimal prediction strategies for different metrics, such as:
- Precision at k
- Propensity-scored precision at k
- Macro-averaged recall at k
- Macro-averaged balanced accuracy at k
- and others ...

### Optimization of prediction for a given test set using Block Coordinate Ascent/Descent (BCA/BCD)

The method aims to optimize the prediction for a given test set using the block coordinate ascent/descent algorithm.

The method was first introduced and described in the paper:
> [Erik Schultheis, Marek Wydmuch, Wojciech Kotłowski, Rohit Babbar, Krzysztof Dembczyński. Generalized test utilities for long-tail performance in extreme multi-label classification. NeurIPS 2023.](https://arxiv.org/abs/2311.05081)

### Finding optimal population classifier via Frank-Wolfe (FW)

The method was first introduced and described in the paper:
> [Erik Schultheis, Wojciech Kotłowski, Marek Wydmuch, Rohit Babbar, Krzysztof Dembczyński. Consistent algorithms for multi-label classification with macro-at-k metrics. ICLR 2024.](https://arxiv.org/abs/2401.16594)


## Repository structure

The repository is organized as follows:
- `docs/` - Sphinx documentation (work in progress)
- `experiments/` - a code for reproducing experiments from the papers, see the README file in the directory for details
- `xcolumns/` - Python package with the library
- `tests/` - some tests for the library (work in progress, the coverage is very limited at the moment)


## Development and contributing

The library was created as a part of our research projects.
We are happy to share it with the community and we hope that someone will find it useful.
If you have any questions or suggestions or if you found a bug, please open an issue.
We are also happy to accept contributions in the form of pull requests.
