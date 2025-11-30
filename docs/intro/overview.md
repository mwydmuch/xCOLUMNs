# Overview of xCOLUMNs library

## What is multi-label classification?

Multi-label classification is a type of classification task, where each instance can be assigned to multiple labels.
For example, a news article can be assigned to multiple categories, such as "politics", "sports", "technology", etc.
The labels for each instance are usually represented as binary vectors $y \in \{0, 1\}^m$, where $m$ is the number of possible labels, and $y_j = 1$ means that label $j$ is relevant for the given instance, and $y_j = 0$ otherwise.


## What are label-wise utilities?

The label-wise utilities are performance metrics that decompose over labels, i.e., they can be defined as a sum of individual metrics for each label:

$$
    \Psi(Y, \hat{Y}) = \sum_{j=1}^{m} \psi^j(y_{:,j}, \hat{y}_{:,j}) \,,
$$

where $Y$ is the true label matrix, $\hat{Y}$ is the predicted label matrix (both of size $n \times m$, where $n$ is number of instances) and $y_{:,j}$ is the $j$-th column of $Y$, and $\psi^j$ is binary utility function for label $j$.

Such metrics, under the assumption that are order-invariant, can be defined as a function of the collection of confusion matrices for each label, i.e., $C_j = \text{confusion matrix}(y_{:,j}, \hat{y}_{:,j})$.
Where confusion matrix $C_j$ is a 2x2 matrix with entries: true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN).

The family of label-wise utilities includes standard instance-wise metrics, such as precision at k or Hamming score/loss, that are popular in extreme classification and recommendation tasks as well as more complex metrics, such as
macro-averaged F-score, Jaccard score, macro-averged precision and recall at k etc.


## Main idea of xCOLUMNs

So far, all the algorithms implemented in xCOLUMNs require providing the definition of the target function and marginal conditional probabilities (or their estimates) of labels - $P(y_{j} = 1 | x)$
for each label $j$ and each instance $x$ resulting in matrix that we denote as $H$ (large $\eta$) and in the names as **`y_proba`** argument of different methods.
The estimates of label probabilities can be easily obtained from many different classification models, such as logistic regression, neural networks, etc.
In this sense xCOLUMNs implements plug-in inference methods, that can be used on top of any probability estimatation model.


## What methods are implemented in xCOLUMNs?

The aim of xCOLUMNs is to provide methods for the optimization of the general family of label-wise utilities. Currently, the following methods are implemented:

- Weighted instance-wise prediction that include optimal infernece strategies for some metrics. Implemented in [`xcolumns.weighted_prediction`](../api/weighted_prediction) module.

- Prediction for provided label-wise metric and test set using **Block Coordinate Ascent/Descent (BC)** method. Implemented in [`xcolumns.block_coordinate`](../api/block_coordinate) module. It was first introduced and described in:
> [Erik Schultheis, Marek Wydmuch, Wojciech Kotłowski, Rohit Babbar, Krzysztof Dembczyński. Generalized test utilities for long-tail performance in extreme multi-label classification. NeurIPS 2023.](https://arxiv.org/abs/2311.05081)

- Search for optimal population classifier for provided metric defined on mulit-label confusion matrix using **Frank-Wolfe (FW)** method and provided training set. Implemented in [`xcolumns.frank_wolfe`](../api/frank_wolfe) module. It was first introduced and described in:
> [Erik Schultheis, Wojciech Kotłowski, Marek Wydmuch, Rohit Babbar, Strom Borman, Krzysztof Dembczyński. Consistent algorithms for multi-label classification with macro-at-k metrics. ICLR 2024.](https://arxiv.org/abs/2401.16594)

The library also implements a set of methods for calculating the metrics based on both the confusion matrix and the true and predicted labels. Implemented in [`xcolumns.confusion_matrix`](../api/confusion_matrix) and [`xcolumns.metrics`](../api/metrics) modules.


## Convetions





## Logging

The library uses the `logging` module for logging. The default logging level is `INFO`. To change the logging level, use the following code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
