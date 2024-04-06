# Metrics

`xcolumns.metrics` module implements a set of methods for calculating the metrics based on both the confusion matrix and the true and predicted labels.
The methods calculating the metrics on the entries of the confusion matrix can be also used as arguments for the methods in the [`xcolumns.block_coordinate`](api/block_coordinate) and [`xcolumns.frank_wolfe`](api/frank_wolfe) modules.

```{eval-rst}
.. automodule:: xcolumns.metrics
   :members:
   :exclude-members: make_macro_metric_on_conf_matrix, make_micro_metric_on_conf_matrix, make_metric_on_y_true_and_y_pred
   :undoc-members:
   :show-inheritance:
```

## Factory functions for specific metrics

The module provides the factory functions for creating the metric micro- and macro-averaged functions based on the binary metric function as well as the functions that calculate the metrics on true and predicted labels.

```{eval-rst}
.. autofunction:: make_macro_metric_on_conf_matrix
.. autofunction:: make_micro_metric_on_conf_matrix
.. autofunction:: make_metric_on_y_true_and_y_pred
```
