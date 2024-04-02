# Weighted predictions

`xcolumns.weighted_prediction` module provides the methods for calculating the weighted prediction for each instance based on the conditional probabilities of labels.
The main function of the module is [**predict_weighted_per_instance**](#xcolumns.weighted_prediction.predict_weighted_per_instance).


```{eval-rst}
.. autofunction:: xcolumns.weighted_prediction.predict_weighted_per_instance
```

---


Based on this function the module provides few additional functions for calculating the predictions
that are optimal for some specific metrics or arbitrary upweight labels with smaller prior probabilities.

```{eval-rst}
.. automodule:: xcolumns.weighted_prediction
   :members:
   :exclude-members: predict_weighted_per_instance
   :undoc-members:
   :show-inheritance:
```
