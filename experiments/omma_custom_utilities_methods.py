import autograd.numpy as anp
import numpy as np
import torch

from xcolumns.metrics import *
from xcolumns.types import *
from xcolumns.utils import *


def comp_log(x):
    if isinstance(x, np.ndarray):
        return np.log(x)
    elif isinstance(x, torch.Tensor):
        return torch.log(x)
    elif isinstance(x, anp.numpy_boxes.ArrayBox):
        return anp.log(x)


def comp_exp(x):
    if isinstance(x, np.ndarray):
        return np.exp(x)
    elif isinstance(x, torch.Tensor):
        return torch.exp(x)
    elif isinstance(x, anp.numpy_boxes.ArrayBox):
        return anp.exp(x)
    else:
        return np.exp(x)


def no_support(y_true):
    return None, {"time": 0, "iters": 0}


def binary_min_tp_tn_on_conf_matrix(tp, fp, fn, tn):
    return np.fmin(tp, tn)


def macro_min_tp_tn_on_conf_matrix(tp, fp, fn, tn):
    if not isinstance(tp, torch.Tensor):
        tp = torch.tensor(tp, dtype=DefaultTorchDataDType)
        tn = torch.tensor(tn, dtype=DefaultTorchDataDType)
    return torch.min(tp, tn).mean()


def macro_min_tp_tn(y_true, y_pred):
    return macro_min_tp_tn_on_conf_matrix(
        *calculate_confusion_matrix(y_true, y_pred, normalize=True)
    )


def multiclass_hmean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
):
    vals = (tp + fn) / (tp + 1e-9)
    vals = vals.sum() ** -1 * vals.shape[0]
    return vals


def multiclass_hmean(y_true, y_pred):
    return multiclass_hmean_on_conf_matrix(*calculate_confusion_matrix(y_true, y_pred))


def multiclass_qmean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
):
    vals = ((fn) / (tp + fn + 1e-9)) ** 2
    vals = vals.sum() / vals.shape[0]
    vals = 1.0 - vals**0.5
    return vals


def multiclass_qmean(y_true, y_pred):
    return multiclass_qmean_on_conf_matrix(*calculate_confusion_matrix(y_true, y_pred))


# import warnings
# warnings.filterwarnings("error")


def log_multiclass_gmean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
):
    vals = (tp) / (tp + fn + 1e-9)
    vals = comp_log(vals)
    vals = vals.sum()
    return vals


def multiclass_gmean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
):
    vals = (tp) / (tp + fn + 1e-9)
    vals = (vals.prod()) ** (1.0 / tp.shape[0])
    return vals


# multiclass_gmean_on_conf_matrix = log_multiclass_gmean_on_conf_matrix
# multiclass_gmean_on_conf_matrix = true_multiclass_gmean_on_conf_matrix


def true_multiclass_gmean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
):
    vals = (tp) / (tp + fn + 1e-9)
    vals = (vals.prod()) ** (1.0 / tp.shape[0])
    return vals


def multiclass_gmean(y_true, y_pred):
    return multiclass_gmean_on_conf_matrix(*calculate_confusion_matrix(y_true, y_pred))


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
