from time import time
from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

from .default_types import *
from .numba_csr_functions import *
from .utils import *


########################################################################################
# General functions for weighted prediction
########################################################################################


def _predict_weighted_per_instance_torch(
    y_proba: torch.tensor,
    k: int,
    th: float = 0.0,
    a: Optional[torch.tensor] = None,
    b: Optional[torch.tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.tensor:
    n, m = y_proba.shape
    y_pred = torch.zeros(
        n, m, dtype=y_proba.dtype if dtype is None else dtype, device=y_proba.device
    )

    gains = y_proba
    if a is not None:
        gains = gains * a
    if b is not None:
        gains = gains + b

    if k > 0:
        _, top_k = torch.topk(gains, k, axis=1)
        y_pred[torch.arange(n)[:, None], top_k] = 1
    else:
        y_pred[gains >= th] = 1

    return y_pred


def _predict_weighted_per_instance_np(
    y_proba: np.ndarray,
    k: int,
    th: float = 0.0,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    n, m = y_proba.shape
    y_pred = np.zeros((n, m), dtype=y_proba.dtype if dtype is None else dtype)

    gains = y_proba
    if a is not None:
        gains = gains * a
    if b is not None:
        gains = gains + b

    if k > 0:
        top_k = np.argpartition(-gains, k, axis=1)[:, :k]
        y_pred[np.arange(n)[:, None], top_k] = 1
    else:
        y_pred[gains >= th] = 1

    return y_pred


def _predict_weighted_per_instance_csr(
    y_proba: csr_matrix,
    k: int,
    th: float = 0.0,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
) -> csr_matrix:
    n, m = y_proba.shape
    (
        y_pred_data,
        y_pred_indices,
        y_pred_indptr,
    ) = numba_predict_weighted_per_instance_csr(
        *unpack_csr_matrix(y_proba), k, th, a, b
    )
    return csr_matrix((y_pred_data, y_pred_indices, y_pred_indptr), shape=(n, m))


def predict_weighted_per_instance(
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    th: float = 0,
    weights: Union[np.ndarray, torch.tensor, None] = None,
    bases: Union[np.ndarray, torch.tensor, None] = None,
    a: Union[np.ndarray, torch.tensor, None] = None,
    b: Union[np.ndarray, torch.tensor, None] = None,
    dtype: Optional[Union[np.dtype, torch.dtype]] = None,
    return_meta: bool = False,
) -> Union[np.ndarray, torch.tensor, csr_matrix]:
    """
    Predict ... TODO: Add description
    """

    # Support for weights and bases aliases
    if weights is not None and a is None:
        a = weights

    if bases is not None and b is None:
        b = bases

    # Arguments validation
    if not isinstance(y_proba, (np.ndarray, torch.tensor, csr_matrix)):
        raise ValueError(
            "y_proba must be either np.ndarray, torch.tensor, or csr_matrix"
        )

    if len(y_proba.shape) == 1:
        y_proba = y_proba.reshape(1, -1)
    elif len(y_proba.shape) > 2:
        raise ValueError("y_proba must be 1d or 2d")

    n, m = y_proba.shape
    if a is not None:
        if not isinstance(a, (np.ndarray, torch.tensor)):
            raise ValueError("a must be np.ndarray or torch.tensor")

        if a.shape != (m,):
            raise ValueError("a must be of shape (y_proba[1],)")

    if b is not None:
        if not isinstance(b, (np.ndarray, torch.tensor)):
            raise ValueError("b must be np.ndarray or torch.tensor")

        if b.shape != (m,):
            raise ValueError("b must be of shape (y_proba[1],)")

    # Initialize the meta data dictionary
    if return_meta:
        meta = {"iters": 1, "time": time()}

    # Invoke the specialized implementation
    if isinstance(y_proba, np.ndarray):
        y_pred = _predict_weighted_per_instance_np(
            y_proba, k, th=th, a=a, b=b, dtype=dtype
        )
    elif isinstance(y_proba, torch.tensor):
        y_pred = _predict_weighted_per_instance_torch(
            y_proba, k, th=th, a=a, b=b, dtype=dtype
        )
    elif isinstance(y_proba, csr_matrix):
        y_pred = _predict_weighted_per_instance_csr(
            y_proba, k, th=th, a=a, b=b, dtype=dtype
        )

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


########################################################################################
# Specialized functions for specific metrics
########################################################################################


def predict_top_k(
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    return_meta: bool = False,
):
    return predict_weighted_per_instance(y_proba, k=k, return_meta=return_meta)


# Implementations of different weighting schemes
def predict_for_optimal_macro_recall(  # (for population)
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    priors: Union[np.ndarray, torch.tensor],
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
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    inv_ps: Union[np.ndarray, torch.tensor],
    return_meta: bool = False,
    **kwargs,
):
    if inv_ps.shape[0] != y_proba.shape[1]:
        raise ValueError("inv_ps must be of shape (y_proba[1],)")

    return predict_weighted_per_instance(
        y_proba, k=k, weights=inv_ps, return_meta=return_meta
    )


def predict_log_weighted_per_instance(
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    priors: Union[np.ndarray, torch.tensor],
    epsilon: float = 1e-6,
    return_meta: bool = False,
    **kwargs,
):
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = -np.log(priors + epsilon)
    return predict_weighted_per_instance(
        y_proba, k=k, weights=weights, return_meta=return_meta
    )


def predict_sqrt_weighted_instance(
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    priors: Union[np.ndarray, torch.tensor],
    epsilon: float = 1e-6,
    return_meta: bool = False,
    **kwargs,
):
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / np.sqrt(priors + epsilon)
    return predict_weighted_per_instance(
        y_proba, k=k, weights=weights, return_meta=return_meta
    )


def predict_power_law_weighted_instance(
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    priors: Union[np.ndarray, torch.tensor],
    beta: float,
    epsilon: float = 1e-6,
    return_meta: bool = False,
    **kwargs,
):
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / (priors + epsilon) ** beta
    return predict_weighted_per_instance(
        y_proba, k=k, weights=weights, return_meta=return_meta
    )


def predict_for_optimal_instance_precision(
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    return_meta: bool = False,
    **kwargs,
):
    return predict_top_k(y_proba, k=k, return_meta=return_meta)


def _predict_for_optimal_macro_balanced_accuracy_np(
    y_proba: np.ndarray,
    k: int,
    priors: Union[np.ndarray, torch.tensor],
    epsilon: float = 1e-6,
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
    y_proba: csr_matrix,
    k: int,
    priors: Union[np.ndarray, torch.tensor],
    epsilon: float = 1e-6,
):
    n, m = y_proba.shape
    assert priors.shape == (m,)
    priors = priors + epsilon

    data, indices, indptr = numba_predict_macro_balanced_accuracy(
        *unpack_csr_matrix(y_proba),
        n,
        m,
        k,
        priors,
    )
    return csr_matrix((data, indices, indptr), shape=y_proba.shape)


def predict_for_optimal_macro_balanced_accuracy(  # (for population)
    y_proba: Union[np.ndarray, torch.tensor, csr_matrix],
    k: int,
    priors: Union[np.ndarray, torch.tensor],
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
        y_pred = _predict_weighted_per_instance_np(y_proba, k, priors, epsilon=epsilon)
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        y_pred = _predict_weighted_per_instance_csr(y_proba, k, priors, epsilon=epsilon)
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred
