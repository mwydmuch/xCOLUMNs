from time import time
from typing import Union, Optional

import numpy as np
from scipy.sparse import csr_matrix

from .default_types import *
from .numba_csr_methods import *
from .utils import *


def _predict_weighted_per_instance_np_fast(y_proba: np.ndarray, k: int, a: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None):
    n, m = y_proba.shape
    y_pred = np.zeros((n, m), dtype=FLOAT_TYPE)
    
    gains = y_proba
    if a is not None:
        gains *= a
    if b is not None:
        gains += b
        
    top_k = np.argpartition(-gains, k, axis=1)[:, :k]
    y_pred[top_k] = 1.0

    return y_pred


def _predict_weighted_per_instance_np(y_proba: np.ndarray, k: int, t: float =0.0, a: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None):
    n, m = y_proba.shape
    y_pred = np.zeros((n, m), dtype=FLOAT_TYPE)

    for i in range(n):
        gains = y_proba[i, :]
        if a is not None:
            gains = gains * a
        if b is not None:
            gains = gains + b

        if k > 0:
            top_k = np.argpartition(-gains, k)[:k]
            y_pred[i, top_k] = 1.0
        else:
            y_pred[i, gains >= t] = 1.0

    return y_pred


def _predict_weighted_per_instance_csr(
    y_proba: csr_matrix, k: int, t: float =0.0, a: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None,
):  
    n, m = y_proba.shape
    y_pred_data, y_pred_indices, y_pred_indptr = numba_predict_weighted_per_instance(
        *unpack_csr_matrix(y_proba), n, m, k, t, a, b  
    )
    return csr_matrix((y_pred_data, y_pred_indices, y_pred_indptr), shape=(n, m))


def predict_weighted_per_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    t: float = 0,
    weights: Optional[np.ndarray] = None,
    bases: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    return_meta: bool =False,
):
    # Support for weights and bases aliases
    if weights is not None and a is None:
        a = weights

    if bases is not None and b is None:
        b = bases

    # Arguments validation
    if not isinstance(y_proba, (np.ndarray, csr_matrix)):
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    n, m = y_proba.shape
    if a is not None:
        if not isinstance(a, np.ndarray):
            raise ValueError("a must be np.ndarray")
        
        if a.shape != (m,):
            raise ValueError("a must be of shape (y_proba[1],)")
        
    if b is not None:
        if not isinstance(b, np.ndarray):
            raise ValueError("b must be np.ndarray")
        
        if b.shape != (m,):
            raise ValueError("b must be of shape (y_proba[1],)")

    # Initialize the meta data dictionary
    if return_meta:
        meta = {"iters": 1, "time": time()}

    # Invoke the specialized implementation
    if isinstance(y_proba, np.ndarray):
        y_pred = _predict_weighted_per_instance_np(y_proba, k, t=t, a=a, b=b)
    elif isinstance(y_proba, csr_matrix):
        y_pred = _predict_weighted_per_instance_csr(y_proba, k, t=t, a=a, b=b)

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


def predict_top_k(y_proba: Union[np.ndarray, csr_matrix], k: int, return_meta=True):
    n, m = y_proba.shape
    return predict_weighted_per_instance(y_proba, k=k, return_meta=return_meta)


# Implementations of different weighting schemes
def predict_for_optimal_macro_recall(  # (for population)
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    priors: np.ndarray,
    epsilon: float = 1e-6,
    return_meta: bool = False,
    **kwargs,
):  
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / (priors + epsilon)
    return predict_weighted_per_instance(
        y_proba, k=k, weights=weights, return_meta=return_meta
    )


def predict_inv_propensity_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    inv_ps: np.ndarray,
    return_meta: bool = False,
    **kwargs,
):
    if inv_ps.shape[0] != y_proba.shape[1]:
        raise ValueError("inv_ps must be of shape (y_proba[1],)")

    return predict_weighted_per_instance(y_proba, k=k, weights=inv_ps, return_meta=return_meta)


def predict_log_weighted_per_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    priors: np.ndarray,
    epsilon: float = 1e-6,
    return_meta: bool = False,
    **kwargs,
):
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = -np.log(priors + epsilon)
    return predict_weighted_per_instance(y_proba, k=k, weights=weights, return_meta=return_meta)


def predict_sqrt_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    priors: np.ndarray,
    epsilon: float = 1e-6,
    return_meta: bool = False,
    **kwargs,
):
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / np.sqrt(priors + epsilon)
    return predict_weighted_per_instance(y_proba, k=k, weights=weights, return_meta=return_meta)


def predict_power_law_weighted_instance(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    priors: np.ndarray,
    beta: float,
    epsilon: float = 1e-6,    
    return_meta: bool = False,
    **kwargs,
):  
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / (priors + epsilon) ** beta
    return predict_weighted_per_instance(y_proba, k=k, weights=weights, return_meta=return_meta)


def predict_for_optimal_instance_precision(
    y_proba: Union[np.ndarray, csr_matrix], k: int, return_meta: bool = False, **kwargs
):
    return predict_top_k(y_proba, k=k, return_meta=return_meta)


def _predict_for_optimal_macro_balanced_accuracy_np(
    y_proba: np.ndarray, k: int, priors: np.ndarray, epsilon: float = 1e-6
):
    n, m = y_proba.shape
    assert priors.shape == (m,)
    priors = priors + epsilon

    y_pred = np.zeros((n, m), np.float32)
    for i in range(n):
        eta = y_proba[i, :]
        g = eta / priors - (1 - eta) / (1 - priors)
        top_k = np.argpartition(-g, k)[:k]
        y_pred[i, top_k] = 1.0

    return y_pred


def _predict_for_optimal_macro_balanced_accuracy_csr(
    y_proba: csr_matrix, k: int, priors: np.ndarray, epsilon: float = 1e-6
):
    n, m = y_proba.shape
    assert priors.shape == (m,)
    priors = priors + epsilon

    data, indices, indptr = numba_predict_macro_balanced_accuracy(
        *unpack_csr_matrix(y_proba), n, m, k, priors,
    )
    return csr_matrix((data, indices, indptr), shape=y_proba.shape)


def predict_for_optimal_macro_balanced_accuracy(  # (for population)
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    priors: np.ndarray,
    epsilon: float = 1e-6,
    return_meta: bool = False,
    **kwargs,
):
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    if return_meta:
        meta = {"iters": 1, "time": time()}

    if isinstance(y_proba, np.ndarray):
        # Invoke original dense implementation
        y_pred = _predict_weighted_per_instance_np(
            y_proba, k, priors, epsilon=epsilon
        )
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        y_pred = _predict_weighted_per_instance_csr(
            y_proba, k, priors, epsilon=epsilon
        )
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred
