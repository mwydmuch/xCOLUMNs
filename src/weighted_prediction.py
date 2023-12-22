import numpy as np
from scipy.sparse import csr_matrix
from utils_sparse import *
from typing import Union
from time import time
from default_types import FLOAT_TYPE, INT_TYPE


MARGINALS_EPS = 1e-6


def predict_weighted_per_instance_np(y_proba: np.ndarray, weights: np.ndarray, k: int):
    ni, nl = y_proba.shape
    assert weights.shape == (nl,)

    result = np.zeros((ni, nl), dtype=FLOAT_TYPE)

    for i in range(ni):
        eta = y_proba[i, :]
        g = eta * weights
        top_k = np.argpartition(-g, k)[:k]
        result[i, top_k] = 1.0
    return result


def predict_weighted_per_instance_csr(y_proba: csr_matrix, weights: np.ndarray, k: int):
    # Since many numpy functions are not supported for sparse matrices
    ni, nl = y_proba.shape
    data, indices, indptr = numba_weighted_per_instance(
        y_proba.data, y_proba.indices, y_proba.indptr, weights, ni, nl, k
    )
    return csr_matrix((data, indices, indptr), shape=y_proba.shape)


def predict_weighted_per_instance(
    y_proba: Union[np.ndarray, csr_matrix], weights: np.ndarray, k: int, return_meta = False
):
    if return_meta:
        meta = {"iters": 1, "time": time()}

    if isinstance(y_proba, np.ndarray):
        # Invoke original dense implementation of Erik
        result = predict_weighted_per_instance_np(y_proba, weights, k=k)
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        result = predict_weighted_per_instance_csr(y_proba, weights, k=k)
    
    if return_meta:
        meta["time"] = time() - meta["time"]
        result = (result, meta)
    
    return result


def predict_top_k(y_proba: Union[np.ndarray, csr_matrix], k: int, return_meta = True):
    ni, nl = y_proba.shape
    weights = np.ones((nl,), dtype=FLOAT_TYPE)
    return predict_weighted_per_instance(y_proba, weights, k=k, return_meta=return_meta)


def predict_random_k(y_proba: Union[np.ndarray, csr_matrix], k: int):
    """
    TODO
    """
    pass
    

# Implementations of different weighting schemes
def predict_for_optimal_macro_recall(  # (for population)
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    return_meta: bool = False,
    **kwargs
):
    return predict_weighted_per_instance(y_proba, 1.0 / (marginals + epsilon), k=k, return_meta=return_meta)


def inv_propensity_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix], k: int, inv_ps: np.ndarray, return_meta: bool = False, **kwargs
):
    return predict_weighted_per_instance(y_proba, inv_ps, k=k, return_meta=return_meta)


def predict_log_weighted_per_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    return_meta: bool = False,
    **kwargs
):
    weights = -np.log(marginals + epsilon)
    return predict_weighted_per_instance(y_proba, weights, k=k, return_meta=return_meta)


def sqrt_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    return_meta: bool = False,
    **kwargs
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
    **kwargs
):
    weights = 1.0 / (marginals + epsilon) ** beta
    return predict_weighted_per_instance(y_proba, weights, k=k, return_meta=return_meta)


def predict_for_optimal_instance_precision(
    y_proba: Union[np.ndarray, csr_matrix], k: int, return_meta: bool = False, **kwargs
):
    return predict_top_k(y_proba, k=k, return_meta=return_meta)


def optimal_macro_balanced_accuracy(  # (for population)
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    marginals: np.ndarray,
    epsilon: float = MARGINALS_EPS,
    return_meta: bool = False,
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
        data, indices, indptr = numba_macro_balanced_accuracy(
            y_proba.data, y_proba.indices, y_proba.indptr, marginals, ni, nl, k
        )
        return csr_matrix((data, indices, indptr), shape=y_proba.shape), {"iters": 1}
