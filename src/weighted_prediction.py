import numpy as np
from scipy.sparse import csr_matrix
from utils_sparse import *
from typing import Union


MARGINALS_EPS = 1e-6


def weighted_per_instance_np(y_proba: np.ndarray, weights: np.ndarray, k: int):
    ni, nl = y_proba.shape
    assert weights.shape == (nl,)

    result = np.zeros((ni, nl), np.float32)

    for i in range(ni):
        eta = y_proba[i, :]
        g = eta * weights
        top_k = np.argpartition(-g, k)[:k]
        result[i, top_k] = 1.0
    return result, {"iters": 1}


def weighted_per_instance_csr(y_proba: csr_matrix, weights: np.ndarray, k: int):
    # Since many numpy functions are not supported for sparse matrices
    ni, nl = y_proba.shape
    data, indices, indptr = numba_weighted_per_instance(
        y_proba.data, y_proba.indices, y_proba.indptr, weights, ni, nl, k
    )
    return csr_matrix((data, indices, indptr), shape=y_proba.shape), {"iters": 1}


def weighted_per_instance(
    y_proba: Union[np.ndarray, csr_matrix], weights: np.ndarray, k: int
):
    if isinstance(y_proba, np.ndarray):
        # Invoke original dense implementation of Erik
        return weighted_per_instance_np(y_proba, weights, k=k)
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        return weighted_per_instance_csr(y_proba, weights, k=k)


# Implementations of different weighting schemes


def optimal_macro_recall(  # (for population)
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    **kwargs
):
    return weighted_per_instance(y_proba, 1.0 / (marginals + epsilon), k=k)


def inv_propensity_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix], k: int, inv_ps: np.ndarray, **kwargs
):
    return weighted_per_instance(y_proba, inv_ps, k=k)


def log_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    **kwargs
):
    weights = -np.log(marginals + epsilon)
    return weighted_per_instance(y_proba, weights, k=k)


def sqrt_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    **kwargs
):
    weights = 1.0 / np.sqrt(marginals + epsilon)
    return weighted_per_instance(y_proba, weights, k=k)


def power_law_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    beta: float = 0.25,
    **kwargs
):
    weights = 1.0 / (marginals + epsilon) ** beta
    return weighted_per_instance(y_proba, weights, k=k)


def optimal_instance_precision(
    y_proba: Union[np.ndarray, csr_matrix], k: int, **kwargs
):
    ni, nl = y_proba.shape
    weights = np.ones((nl,), dtype=np.float32)
    return weighted_per_instance(y_proba, weights, k=k)


def optimal_balanced_accuracy(  # (for population)
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    **kwargs
):
    ni, nl = y_proba.shape
    assert marginals.shape == (nl,)
    marginals = marginals + epsilon

    if isinstance(y_proba, np.ndarray):
        result = np.zeros((ni, nl), np.float32)
        for i in range(ni):
            eta = y_proba[i, :]
            g = eta / marginals - (1 - eta) / (1 - marginals)
            top_k = np.argpartition(-g, k)[:k]
            result[i, top_k] = 1.0
        return result, {"iters": 1}
    
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        data, indices, indptr = numba_balanced_accuracy(
            y_proba.data, y_proba.indices, y_proba.indptr, marginals, ni, nl, k
        )
        return csr_matrix((data, indices, indptr), shape=y_proba.shape), {"iters": 1}
