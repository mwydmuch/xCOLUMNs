# Block Coordinate-based prediction methods (`xcolumns.block_coordinate`)

`xcolumns.block_coordinate` module implements the methods for finding the optimal prediction for given test set using the Block Coordinate Ascend/Desend algorithm with 0-th order approximation of expected utility.
The method was first introduced and described in the paper:
> [Erik Schultheis, Marek Wydmuch, Wojciech Kotłowski, Rohit Babbar, Krzysztof Dembczyński. Generalized test utilities for long-tail performance in extreme multi-label classification. NeurIPS 2023.](https://arxiv.org/abs/2311.05081)

Note: BCA/BCD with 0-approximationuses tp, fp, fn, tn matrices parametrization of the confussion matrix,
as opposed to algorithms presented in the paper, which use :math:`t, q, p` parametrization. However both algorithms are equivalent.

The main function of the module is {func}`predict_using_bc_with_0approx() <xcolumns.block_coordinate.predict_using_bc_with_0approx>`:

```{eval-rst}
.. autofunction:: xcolumns.block_coordinate.predict_using_bc_with_0approx
```

## Wrapper functions for specific metrics

The module provides the wrapper functions for specific metrics that can be used as arguments for the {func}`predict_using_bc_with_0approx() <xcolumns.block_coordinate.predict_using_bc_with_0approx>` function as well as factory function for creating such wrapper functions.

```{eval-rst}
.. automodule:: xcolumns.block_coordinate
   :members:
   :exclude-members: predict_using_bc_with_0approx, predict_optimizing_coverage_using_bc
   :undoc-members:
   :show-inheritance:
```


## Special function for optimization of coverage

The module provides the special function for optimization of coverage metric that use other way of estimating the expected value of the metric than {func}`predict_using_bc_with_0approx() <xcolumns.block_coordinate.predict_using_bc_with_0approx>` function.

```{eval-rst}
.. autofunction:: xcolumns.block_coordinate.predict_optimizing_coverage_using_bc
```
