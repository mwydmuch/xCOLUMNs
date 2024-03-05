from time import time
from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .numba_csr_functions import *
from .types import *
from .utils import *


########################################################################################
# General functions for weighted prediction
########################################################################################

_torch_available = None
try:
    # Try to import torch
    import torch

    # Define the specialized implementation for torch.Tensor
    def _predict_weighted_per_instance_torch(
        y_proba: torch.Tensor,
        k: int,
        th: float = 0.0,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
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

    _torch_available = True
except ImportError:
    pass


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
    dtype: Optional[np.dtype] = None,
) -> csr_matrix:
    if a is not None and a.dtype != y_proba.dtype:
        a = a.astype(y_proba.dtype)
    if b is not None and b.dtype != y_proba.dtype:
        b = b.astype(y_proba.dtype)

    n, m = y_proba.shape
    (
        y_pred_data,
        y_pred_indices,
        y_pred_indptr,
    ) = numba_predict_weighted_per_instance_csr(
        *unpack_csr_matrix(y_proba), k, th, a, b
    )
    return csr_matrix(
        (y_pred_data, y_pred_indices, y_pred_indptr), shape=(n, m), dtype=dtype
    )


def predict_weighted_per_instance(
    y_proba: Matrix,
    k: int,
    th: float = 0.0,
    a: Optional[DenseMatrix] = None,
    b: Optional[DenseMatrix] = None,
    dtype: Optional[DType] = None,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    """
    Predict ... TODO: Add description
    """

    # Arguments validation

    # Check y_proba
    if not isinstance(y_proba, Matrix):
        raise ValueError(
            "y_proba must be either np.ndarray, torch.Tensor, or csr_matrix"
        )

    if len(y_proba.shape) == 1:
        y_proba = y_proba.reshape(1, -1)
    elif len(y_proba.shape) > 2:
        raise ValueError("y_proba must be 1d or 2d")

    # Check k and th
    if not isinstance(k, int):
        raise ValueError("k must be an integer")

    # Check a and b
    n, m = y_proba.shape
    if a is not None:
        if not isinstance(a, DenseMatrix):
            raise ValueError("a must be np.ndarray or torch.Tensor")

        if a.shape != (m,):
            raise ValueError("a must be of shape (y_proba[1],)")

    if b is not None:
        if not isinstance(b, DenseMatrix):
            raise ValueError("b must be np.ndarray or torch.Tensor")

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
    elif isinstance(y_proba, csr_matrix):
        y_pred = _predict_weighted_per_instance_csr(
            y_proba, k, th=th, a=a, b=b, dtype=dtype
        )
    elif _torch_available and isinstance(y_proba, torch.Tensor):
        y_pred = _predict_weighted_per_instance_torch(
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
    y_proba: Matrix,
    k: int,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    return predict_weighted_per_instance(y_proba, k=k, return_meta=return_meta)


# Implementations of different weighting schemes
def predict_for_optimal_macro_recall(  # (for population)
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / (priors + epsilon)
    return predict_weighted_per_instance(
        y_proba, k=k, a=weights, return_meta=return_meta
    )


def predict_inv_propensity_weighted_instance(
    y_proba: Matrix,
    k: int,
    inv_ps: DenseMatrix,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    if inv_ps.shape[0] != y_proba.shape[1]:
        raise ValueError("inv_ps must be of shape (y_proba[1],)")

    return predict_weighted_per_instance(
        y_proba, k=k, a=inv_ps, return_meta=return_meta
    )


def predict_log_weighted_per_instance(
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = -np.log(priors + epsilon)
    return predict_weighted_per_instance(
        y_proba, k=k, a=weights, return_meta=return_meta
    )


def predict_sqrt_weighted_instance(
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / np.sqrt(priors + epsilon)
    return predict_weighted_per_instance(
        y_proba, k=k, a=weights, return_meta=return_meta
    )


def predict_power_law_weighted_instance(
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    beta: float,
    epsilon: float = 1e-6,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / (priors + epsilon) ** beta
    return predict_weighted_per_instance(
        y_proba, k=k, a=weights, return_meta=return_meta
    )


def predict_for_optimal_instance_precision(
    y_proba: Matrix,
    k: int,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    return predict_top_k(y_proba, k=k, return_meta=return_meta)


def _predict_for_optimal_macro_balanced_accuracy_np(
    y_proba: np.ndarray,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
) -> Matrix:
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
    priors: DenseMatrix,
    epsilon: float = 1e-6,
) -> Matrix:
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
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    if return_meta:
        meta = {"iters": 1, "time": time()}

    if isinstance(y_proba, np.ndarray):
        y_pred = _predict_for_optimal_macro_balanced_accuracy_np(
            y_proba, k, priors, epsilon=epsilon
        )
    elif isinstance(y_proba, csr_matrix):
        y_pred = _predict_for_optimal_macro_balanced_accuracy_csr(
            y_proba, k, priors, epsilon=epsilon
        )
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred
