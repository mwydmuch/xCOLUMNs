from typing import Callable, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .default_types import *
from .numba_csr_methods import *


def unpack_csr_matrix(matrix: csr_matrix):
    return matrix.data, matrix.indices, matrix.indptr  # , matrix.shape


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
