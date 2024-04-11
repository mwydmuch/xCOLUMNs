---
hide-toc: true
firstpage:
lastpage:
---

<p align="center">
  <img src="_static/xCOLUMNs_logo.png" width="500px"/>
</p>

# Welcome to xCOLUMNs documentation!

xCOLUMNs stands for x**Consistent Optimization of Label-wise Utilities in Multi-label classificatioN**s.
It is a small Python library that aims to implement different methods for the optimization of a general family of
metrics that can be defined on multi-label classification matrices.
These include, but are not limited to, label-wise metrics.
The library provides an efficient implementation of the different optimization methods
that easily scale to the extreme multi-label classification (XMLC) - problems with a very large number of labels and instances.


```{toctree}
:hidden:
:caption: Introduction

intro/overview
intro/quick_start
```

```{toctree}
:hidden:
:caption: API

api/weighted_prediction
api/block_coordinate
api/frank_wolfe
api/confusion_matrix
api/metrics
```
