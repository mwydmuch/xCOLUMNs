from numbers import Number
from typing import Callable, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .default_types import *
from .numba_csr_methods import *
from .utils import *


def _calculate_tp_np(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sum(y_true * y_pred, axis=0)


# Alternative version, performance is similar
# def _calculate_tp_csr(y_true: csr_matrix, y_pred: csr_matrix):
#     return (y_pred.multiply(y_true)).sum(axis=0)


def _calculate_tp_csr(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    return numba_calculate_sum_0_sparse_mat_mul_mat(
        *unpack_csr_matrices(y_pred, y_true), n, m
    )


def _calculate_fp_np(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sum((1 - y_true) * y_pred, axis=0)


def _calculate_fp_csr_slow(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    fp = np.zeros(m, dtype=FLOAT_TYPE)
    dense_ones = np.ones(m, dtype=FLOAT_TYPE)
    for i in range(n):
        fp += y_pred[i].multiply(dense_ones - y_true[i])
    return fp


def _calculate_fn_np(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sum(y_true * (1 - y_pred), axis=0)


def _calculate_fp_csr(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    return numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(
        *unpack_csr_matrices(y_pred, y_true), n, m
    )


def _calculate_fn_csr_slow(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    fn = np.zeros(m, dtype=FLOAT_TYPE)
    dense_ones = np.ones(m, dtype=FLOAT_TYPE)
    for i in range(n):
        fn += y_true[i].multiply(dense_ones - y_pred[i])

    return fn


def _calculate_fn_csr(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    return numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(
        *unpack_csr_matrices(y_true, y_pred), n, m
    )


def _calculate_tn_np(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sum((1 - y_true) * (1 - y_pred), axis=0)


def _calculate_conf_mat_entry(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    func_for_np: Callable,
    func_for_csr: Callable,
    normalize: bool = False,
):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        val = func_for_np(y_true, y_pred)
    elif isinstance(y_true, csr_matrix) and isinstance(y_pred, csr_matrix):
        val = func_for_csr(y_true, y_pred)
    else:
        raise ValueError("y_true and y_pred must be both dense or both sparse")

    if normalize:
        val /= y_true.shape[0]

    return val


def calculate_tp(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    normalize: bool = False,
):
    return _calculate_conf_mat_entry(
        y_true, y_pred, _calculate_tp_np, _calculate_tp_csr, normalize=normalize
    )


def calculate_fp(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    normalize: bool = False,
):
    return _calculate_conf_mat_entry(
        y_true, y_pred, _calculate_fp_np, _calculate_fp_csr, normalize=normalize
    )


def calculate_fn(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    normalize: bool = False,
):
    return _calculate_conf_mat_entry(
        y_true, y_pred, _calculate_fn_np, _calculate_fn_csr, normalize=normalize
    )


def calculate_confusion_matrix(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    normalize: bool = False,
    skip_tn: bool = False,
):
    """
    Calculate confusion matrix for true and prediction.
    """

    tp = calculate_tp(y_true, y_pred, normalize=normalize)
    fp = calculate_fp(y_true, y_pred, normalize=normalize)
    fn = calculate_fn(y_true, y_pred, normalize=normalize)

    n, m = y_true.shape

    if skip_tn:
        tn = np.zeros(m, dtype=FLOAT_TYPE)
    else:
        tn = np.full(m, 1 if normalize else n, dtype=FLOAT_TYPE) - tp - fp - fn

    return tp, fp, fn, tn


def bin_precision_at_k_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
    k: int,
):
    return tp / k


def bin_precision_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray, None],
    tn: Union[Number, np.ndarray, None],
    epsilon: float = 1e-6,
):
    return tp / (tp + fp + epsilon)


def bin_recall_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray, None],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray, None],
    epsilon: float = 1e-6,
):
    return tp / (tp + fn + epsilon)


def bin_fmeasure_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray, None],
    beta: float = 1.0,
    epsilon: float = 1e-6,
):
    return (1 + beta**2) * tp / ((beta**2 * (tp + fp)) + tp + fn + epsilon)


# Alternative definition of F-measure used in some old experiments
# def bin_fmeasure_on_conf_matrix(tp, fp, fn, tn, beta=1.0, epsilon=1e-6):
#     precision = bin_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
#     recall = bin_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
#     return (
#         (1 + beta**2)
#         * precision
#         * recall
#         / (beta**2 * precision + recall + epsilon)
#     )


def macro_precision_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: Union[np.ndarray, None],
    tn: Union[np.ndarray, None],
    epsilon: float = 1e-6,
):
    return bin_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon).mean()


def macro_recall_on_conf_matrix(
    tp: np.ndarray,
    fp: Union[np.ndarray, None],
    fn: np.ndarray,
    tn: Union[np.ndarray, None],
    epsilon: float = 1e-6,
):
    return bin_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon).mean()


def macro_fmeasure_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: Union[np.ndarray, None],
    beta: float = 1.0,
    epsilon: float = 1e-6,
):
    return bin_fmeasure_on_conf_matrix(
        tp, fp, fn, tn, beta=beta, epsilon=epsilon
    ).mean()


def coverage_on_conf_matrix(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: Union[np.ndarray, None]
):
    pass
