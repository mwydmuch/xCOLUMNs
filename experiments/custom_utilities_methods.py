import numpy as np
import torch

from xcolumns.block_coordinate import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics import *
from xcolumns.types import *
from xcolumns.utils import *
from xcolumns.weighted_prediction import *


def no_support(y_true):
    return None, {"time": 0, "iters": 0}


def binary_min_tp_tn_on_conf_matrix(tp, fp, fn, tn):
    return np.fmin(tp, tn)


def macro_min_tp_tn_on_conf_matrix(tp, fp, fn, tn):
    if not isinstance(tp, torch.Tensor):
        tp = torch.tensor(tp, dtype=TORCH_FLOAT_TYPE)
        tn = torch.tensor(tn, dtype=TORCH_FLOAT_TYPE)
    return torch.min(tp, tn).mean()


def macro_min_tp_tn(y_true, y_pred):
    return macro_min_tp_tn_on_conf_matrix(
        *calculate_confusion_matrix(y_true, y_pred, normalize=True)
    )


def multi_class_hmean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
):
    tpr = (tp + fn) / (tp + 1e-6)
    return tpr.shape[0] / tpr.sum()


def multi_class_hmean(y_true, y_pred):
    return multi_class_hmean_on_conf_matrix(*calculate_confusion_matrix(y_true, y_pred))


def multi_class_gmean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
):
    tpr = tp / (tp + fn + 1e-6)
    return tpr.prod() ** (1 / tpr.shape[0])


def multi_class_gmean(y_true, y_pred):
    return multi_class_gmean_on_conf_matrix(*calculate_confusion_matrix(y_true, y_pred))


def bc_macro_min_tp_tn(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iters: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    n, m = y_proba.shape

    return bc_with_0approx(
        y_proba,
        binary_utility_func=binary_min_tp_tn_on_conf_matrix,
        k=k,
        utility_aggregation="mean",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iters=max_iters,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


def bc_micro_f1(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iters: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    raise ValueError("this wont work")


def bc_macro_hmean(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iters: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    n, m = y_proba.shape

    return bc_with_0approx(
        y_proba,
        binary_utility_func=binary_hmean_on_conf_matrix,
        k=k,
        utility_aggregation="mean",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iters=max_iters,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


def bc_macro_gmean(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iters: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    n, m = y_proba.shape

    return bc_with_0approx(
        y_proba,
        binary_utility_func=binary_gmean_on_conf_matrix,
        k=k,
        utility_aggregation="mean",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iters=max_iters,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )
