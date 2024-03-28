import numpy as np

from xcolumns.block_coordinate import *
from xcolumns.metrics import *
from xcolumns.weighted_prediction import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics_on_conf_matrix import *
from xcolumns.default_types import *
from xcolumns.utils import *


def bin_min_tp_tn(tp, fp, fn, tn):
    return np.fmin(tp, tn)


def bc_macro_min_tp_tn(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iter: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    n, m = y_proba.shape

    return bc_with_0approx(
        y_proba,
        bin_utility_func=bin_min_tp_tn,
        k=k,
        utility_aggregation="mean",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


