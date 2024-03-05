import logging
from typing import Callable, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .numba_csr_functions import *
from .types import *


########################################################################################
# Logger
########################################################################################


logger = logging.getLogger("xcolumns")


def log(msg: str, verbose: bool, level: int = logging.INFO):
    if verbose:
        logger.log(level, msg)


def log_info(msg: str, verbose: bool):
    log(msg, verbose, level=logging.INFO)


def log_debug(msg: str, verbose: bool):
    log(msg, verbose, level=logging.DEBUG)


def log_warning(msg: str, verbose: bool):
    log(msg, verbose, level=logging.WARNING)


def log_error(msg: str, verbose: bool):
    log(msg, verbose, level=logging.ERROR)


########################################################################################
# Functions for generating matrices with random prediction at k
########################################################################################


def random_at_k_np(shape: Tuple[int, int], k: int, seed: int = None) -> np.ndarray:
    n, m = shape
    y_pred = np.zeros(shape, dtype=FLOAT_TYPE)

    rng = np.random.default_rng(seed)
    labels_range = np.arange(m)
    for i in range(n):
        y_pred[i, rng.choice(labels_range, k, replace=False, shuffle=False)] = 1.0
    return y_pred


def random_at_k_csr(shape: Tuple[int, int], k: int, seed: int = None) -> csr_matrix:
    n, m = shape
    y_pred_data, y_pred_indices, y_pred_indptr = numba_random_at_k(n, m, k, seed=seed)
    return construct_csr_matrix(
        y_pred_data,
        y_pred_indices,
        y_pred_indptr,
        shape=shape,
        sort_indices=True,
    )


########################################################################################
# csr_matrix utilities
########################################################################################


def unpack_csr_matrix(matrix: csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return matrix.data, matrix.indices, matrix.indptr  # , matrix.shape


def unpack_csr_matrices(*matrices) -> List[np.ndarray]:
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
) -> csr_matrix:
    mat = csr_matrix((data, indices, indptr), dtype=dtype, shape=shape)
    if sort_indices:
        mat.sort_indices()
    return mat


########################################################################################
# Maximum search functions
########################################################################################


def uniform_search(
    low: float, high: float, step: float, func: Callable
) -> Tuple[float, float]:
    best = low
    best_val = func(low)
    for i in np.arange(low + step, high, step):
        score = func(i)
        if score > best_val:
            best = i
            best_val = score
    return best, best_val


def ternary_search(
    low: float, high: float, eps: float, func: Callable
) -> Tuple[float, float]:
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
