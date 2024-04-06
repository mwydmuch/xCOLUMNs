# Finding population classifiers using Frank Wolfe-based method

`xcolumns.frank_wolfe` module implements the methods for finding the optimal population classifier using the Frank-Wolfe algorithm.
The method was first introduced and described in the paper:
> [Erik Schultheis, Wojciech Kotłowski, Marek Wydmuch, Rohit Babbar, Strom Borman, Krzysztof Dembczyński. Consistent algorithms for multi-label classification with macro-at-k metrics. ICLR 2024.](https://arxiv.org/abs/2401.16594)

The main function of the module is [**find_classifier_using_fw**](#xcolumns.frank_wolfe.find_classifier_using_fw):


```{eval-rst}
.. autofunction:: xcolumns.frank_wolfe.find_classifier_using_fw
```


The function returns the RandomizedWeightedClassifier object that can be used for prediction.
The RandomizedWeightedClassifier is a set of weighted classifiers as defined in
The module also provides the function [**predict_using_randomized_weighted_classifier**](#xcolumns.frank_wolfe.predict_using_randomized_weighted_classifier) for predicting the labels using the RandomizedWeightedClassifier object.


```{eval-rst}
.. autoclass:: xcolumns.frank_wolfe.RandomizedWeightedClassifier
```

```{eval-rst}
.. autofunction:: xcolumns.frank_wolfe.predict_using_randomized_weighted_classifier
```


## Wrapper functions for specific metrics

The module provides the wrapper functions for specific metrics that can be used as arguments for the `find_classifier_using_fw` function as well as factory function for creating such wrapper functions.

```{eval-rst}
.. automodule:: xcolumns.frank_wolfe
   :members:
   :exclude-members: find_classifier_using_fw, RandomizedWeightedClassifier, predict_using_randomized_weighted_classifier
   :undoc-members:
   :show-inheritance:
```
