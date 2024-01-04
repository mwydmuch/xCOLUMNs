from time import time
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix

from .default_types import *
from .numba_csr_methods import *


MARGINALS_EPS = 1e-6


def _predict_weighted_per_instance_np(y_proba: np.ndarray, weights: np.ndarray, k: int):
    n, m = y_proba.shape
    assert weights.shape == (m,)

    y_pred = np.zeros((n, m), dtype=FLOAT_TYPE)
    for i in range(n):
        eta = y_proba[i, :]
        g = eta * weights
        top_k = np.argpartition(-g, k)[:k]
        y_pred[i, top_k] = 1.0
    return y_pred


def _predict_weighted_per_instance_csr(
    y_proba: csr_matrix, weights: np.ndarray, k: int
):
    # Since many numpy functions are not supported for sparse matrices
    n, m = y_proba.shape
    assert weights.shape == (m,)
    data, indices, indptr = numba_weighted_per_instance(
        y_proba.data, y_proba.indices, y_proba.indptr, weights, n, m, k
    )
    return csr_matrix((data, indices, indptr), shape=y_proba.shape)


def predict_weighted_per_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    weights: np.ndarray,
    k: int,
    return_meta=False,
):
    if return_meta:
        meta = {"iters": 1, "time": time()}

    if isinstance(y_proba, np.ndarray):
        # Invoke original dense implementation of Erik
        y_pred = _predict_weighted_per_instance_np(y_proba, weights, k)
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        y_pred = _predict_weighted_per_instance_csr(y_proba, weights, k)
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


def predict_top_k(y_proba: Union[np.ndarray, csr_matrix], k: int, return_meta=True):
    n, m = y_proba.shape
    weights = np.ones((m,), dtype=FLOAT_TYPE)
    return predict_weighted_per_instance(y_proba, weights, k=k, return_meta=return_meta)


# Implementations of different weighting schemes
def predict_for_optimal_macro_recall(  # (for population)
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    return_meta: bool = False,
    **kwargs,
):
    return predict_weighted_per_instance(
        y_proba, 1.0 / (marginals + epsilon), k=k, return_meta=return_meta
    )


def inv_propensity_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    inv_ps: np.ndarray,
    return_meta: bool = False,
    **kwargs,
):
    return predict_weighted_per_instance(y_proba, inv_ps, k=k, return_meta=return_meta)


def predict_log_weighted_per_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    return_meta: bool = False,
    **kwargs,
):
    weights = -np.log(marginals + epsilon)
    return predict_weighted_per_instance(y_proba, weights, k=k, return_meta=return_meta)


def sqrt_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    return_meta: bool = False,
    **kwargs,
):
    weights = 1.0 / np.sqrt(marginals + epsilon)
    return predict_weighted_per_instance(y_proba, weights, k=k, return_meta=return_meta)


def power_law_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    beta: float = 0.25,
    return_meta: bool = False,
    **kwargs,
):
    weights = 1.0 / (marginals + epsilon) ** beta
    return predict_weighted_per_instance(y_proba, weights, k=k, return_meta=return_meta)


def predict_for_optimal_instance_precision(
    y_proba: Union[np.ndarray, csr_matrix], k: int, return_meta: bool = False, **kwargs
):
    return predict_top_k(y_proba, k=k, return_meta=return_meta)


def _predict_for_optimal_macro_balanced_accuracy_np(
    y_proba: np.ndarray, k: int, marginals: np.ndarray, epsilon: float = MARGINALS_EPS
):
    n, m = y_proba.shape
    assert marginals.shape == (m,)
    marginals = marginals + epsilon

    y_pred = np.zeros((n, m), np.float32)
    for i in range(n):
        eta = y_proba[i, :]
        g = eta / marginals - (1 - eta) / (1 - marginals)
        top_k = np.argpartition(-g, k)[:k]
        y_pred[i, top_k] = 1.0

    return y_pred


def _predict_for_optimal_macro_balanced_accuracy_csr(
    y_proba: csr_matrix, k: int, marginals: np.ndarray, epsilon: float = MARGINALS_EPS
):
    n, m = y_proba.shape
    assert marginals.shape == (m,)
    marginals = marginals + epsilon

    data, indices, indptr = numba_macro_balanced_accuracy(
        y_proba.data, y_proba.indices, y_proba.indptr, marginals, n, m, k
    )
    return csr_matrix((data, indices, indptr), shape=y_proba.shape)


def predict_for_optimal_macro_balanced_accuracy(  # (for population)
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    return_meta: bool = False,
    **kwargs,
):
    if return_meta:
        meta = {"iters": 1, "time": time()}

    if isinstance(y_proba, np.ndarray):
        # Invoke original dense implementation
        y_pred = _predict_weighted_per_instance_np(
            y_proba, k, marginals, epsilon=epsilon
        )
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        y_pred = _predict_weighted_per_instance_csr(
            y_proba, k, marginals, epsilon=epsilon
        )
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred
