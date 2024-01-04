from typing import Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .default_types import *
from .numba_csr_methods import *


def unpack_csr_matrix(matrix: csr_matrix):
    return matrix.data, matrix.indices, matrix.indptr


def unpack_csr_matrices(*matrices):
    y_pred = []
    for m in matrices:
        y_pred.extend(unpack_csr_matrix(m))
    return y_pred


def construct_csr_matrix(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    dtype=None,
    shape=None,
    sort_indices=False,
):
    mat = csr_matrix((data, indices, indptr), dtype=dtype, shape=shape)
    if sort_indices:
        mat.sort_indices()
    return mat


def calculate_tp_csr_slow(y_true: csr_matrix, y_pred: csr_matrix):
    return (y_pred.multiply(y_true)).sum(axis=0)


def calculate_tp_csr(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    Etp = np.zeros(m, dtype=FLOAT_TYPE)
    for i in range(n):
        r_start, r_end = y_pred.indptr[i], y_pred.indptr[i + 1]
        p_start, p_end = y_true.indptr[i], y_true.indptr[i + 1]

        data, indices = numba_sparse_vec_mul_vec(
            y_pred.data[r_start:r_end],
            y_pred.indices[r_start:r_end],
            y_true.data[p_start:p_end],
            y_true.indices[p_start:p_end],
        )
        Etp[indices] += data

    return Etp


def calculate_fp_csr_slow(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    Efp = np.zeros(m, dtype=FLOAT_TYPE)
    dense_ones = np.ones(m, dtype=FLOAT_TYPE)
    for i in range(n):
        Efp += y_pred[i].multiply(dense_ones - y_true[i])
    return Efp


def calculate_fp_csr(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    return numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(
        *unpack_csr_matrices(y_pred, y_true), n, m
    )


def calculate_fn_csr_slow(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    Efn = np.zeros(m, dtype=FLOAT_TYPE)
    dense_ones = np.ones(m, dtype=FLOAT_TYPE)
    for i in range(n):
        Efn += y_true[i].multiply(dense_ones - y_pred[i])

    return Efn


def calculate_fn_csr(y_true: csr_matrix, y_pred: csr_matrix):
    n, m = y_true.shape
    return numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(
        *unpack_csr_matrices(y_true, y_pred), n, m
    )


def calculate_confusion_matrix(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    normalize: bool = False,
):
    """
    Calculate confusion matrix for true and prediction.
    """
    assert y_true.shape == y_pred.shape
    n, m = y_true.shape

    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        tp = np.sum(y_true * y_pred, axis=0)
        fp = np.sum((1 - y_true) * y_pred, axis=0)
        fn = np.sum(y_true * (1 - y_pred), axis=0)

    elif isinstance(y_true, csr_matrix) and isinstance(y_pred, csr_matrix):
        tp = calculate_tp_csr(y_true, y_pred)
        fp = calculate_fp_csr(y_true, y_pred)
        fn = calculate_fn_csr(y_true, y_pred)

    else:
        raise ValueError("y_true and y_pred must be both dense or both sparse")

    tn = np.full(m, n, dtype=FLOAT_TYPE) - tp - fp - fn
    if normalize:
        tp /= n
        fp /= n
        fn /= n
        tn /= n

    return tp, fp, fn, tn


def random_at_k_csr(shape: Tuple[int, int], k: int, seed: int = None):
    n, m = shape
    y_pred_data, y_pred_indices, y_pred_indptr = numba_random_at_k(n, m, k, seed=seed)
    return construct_csr_matrix(
        y_pred_data,
        y_pred_indices,
        y_pred_indptr,
        shape=shape,
        sort_indices=True,
    )


def random_at_k_np(shape: Tuple[int, int], k: int, seed: int = None):
    n, m = shape
    y_pred = np.zeros(shape, dtype=FLOAT_TYPE)

    rng = np.random.default_rng(seed)
    labels_range = np.arange(m)
    for i in range(n):
        y_pred[i, rng.choice(labels_range, k, replace=False, shuffle=False)] = 1.0
    return y_pred


def lin_search(low, high, step, func) -> Tuple[float, float]:
    best = low
    best_val = func(low)
    for i in np.arange(low + step, high, step):
        score = func(i)
        if score > best_val:
            best = i
            best_val = score
    return best, best_val


def bin_search(low, high, eps, func) -> Tuple[float, float]:
    while high - low > eps:
        mid = (low + high) / 2
        mid_next = (mid + high) / 2

        if func(mid) < func(mid_next):
            high = mid_next
        else:
            low = mid

    best = (low + high) / 2
    best_val = func(best)
    return best, best_val


def ternary_search(low, high, eps, func) -> Tuple[float, float]:
    while high - low > eps:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        if func(mid1) < func(mid2):
            high = mid2
        else:
            low = mid1

    best = (low + high) / 2
    best_val = func(best)
    return best, best_val
