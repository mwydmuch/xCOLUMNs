# COLUMNs - Consistent Optimization of Label-wise Utilities in Multi-label clasifficatioN

This is a small Python library aims to implement different methods for optimization of general family of label-wise utilities (performance metrics) in multi-label classification.


## Installation

The library can be installed using pip:
```sh
pip install columns
```
It should work on all major platforms (Linux, Windows, Mac) and with Python 3.8+.


## Example

```python
from columns.block_coordinate import bc_with_0approx, bin_f1_score


Y_pred = bc_with_0approx(Y_proba, bin_f1_score)



```
