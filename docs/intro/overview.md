# Overview of xCOLUMNs library

xCOLUMNs stands for x**Consistent Optimization of Label-wise Utilities in Multi-label classificatioN**s.
It is a small Python library that aims to implement different methods for the optimization of a general family of
metrics that can be defined on multi-label classification matrices.
These include, but are not limited to, label-wise metrics (see below for details).
The library provides an efficient implementation of the different optimization methods that easily scale to the extreme multi-label classification (XMLC) - problems with a very large number of labels and instances.


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
In this sense xCOLUMNs implements plug-in inference methods, that can be used on top of any probablity estimatation model.


## What methods are implemented in xCOLUMNs?

The aim of xCOLUMNs is to provide methods for the optimization of the general family of label-wise utilities. Currently, the following methods are implemented:

- Prediction for provided test set using **Block Coordinate Ascent/Descent (BC)** method, described in [1].
- Search for optimal population classifier using **Frank-Wolfe (FW)** method, described in [2].

[1] [Erik Schultheis, Marek Wydmuch, Wojciech Kotłowski, Rohit Babbar, Krzysztof Dembczyński. Generalized test utilities for long-tail performance in extreme multi-label classification. NeurIPS 2023.](https://arxiv.org/abs/2311.05081)

[2] [Erik Schultheis, Wojciech Kotłowski, Marek Wydmuch, Rohit Babbar, Strom Borman, Krzysztof Dembczyński. Consistent algorithms for multi-label classification with macro-at-k metrics. ICLR 2024.](https://arxiv.org/abs/2401.16594)
