from time import time
from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .numba_csr_functions import (
    numba_predict_macro_balanced_accuracy_csr,
    numba_predict_weighted_per_instance_csr,
)
from .types import TORCH_AVAILABLE, DenseMatrix, DType, Matrix
from .utils import unpack_csr_matrix, zeros_like


if TORCH_AVAILABLE:
    import torch


########################################################################################
# General functions for weighted prediction
########################################################################################


def _predict_weighted_per_instance_dense(
    y_proba: DenseMatrix,
    k: int,
    th: float = 0.0,
    a: Optional[DenseMatrix] = None,
    b: Optional[DenseMatrix] = None,
    dtype: Optional[DType] = None,
) -> DenseMatrix:
    n, m = y_proba.shape
    y_pred = zeros_like(y_proba, dtype=dtype)

    gains = y_proba
    if a is not None:
        gains = gains * a
    if b is not None:
        gains = gains + b

    if k > 0:
        # Numpy implementation
        if isinstance(y_proba, np.ndarray):
            top_k = np.argpartition(-gains, k, axis=1)[:, :k]
            y_pred[np.arange(n)[:, None], top_k] = 1

        # Torch implementation
        elif TORCH_AVAILABLE and isinstance(y_proba, torch.Tensor):
            _, top_k = torch.topk(gains, k, axis=1)
            y_pred[torch.arange(n)[:, None], top_k] = 1
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
    r"""
    Returns the weighted prediction for each instance (row) in a provided
    matrix of conditional probabilities estimates of labels :math:`\eta` (**y_proba**).
    The gains vector is calculated according to:

    .. math::
        g = a \cdot \eta + b

    If **k** is larger than 0, the top **k** labels with the highest gains are selected for each instance.
    If **k** is 0, then the labels with gains higher than **th** are selected for each instance.

    Args:
        y_proba: A 2D matrix of conditional probabilities for each label.
        k: The number of labels to predict for each instance.
        th: The threshold for the gains.
        a: The weights for the gains.
        b: The biases for the gains.
        dtype: The data type for the output matrix, if equal to None, the data type of **y_proba** will be used.
        return_meta: Whether to return meta data.

    Returns:
        The prediction matrix, the shape and type of the matrix is the same as **y_proba**. If **return_meta** is True, additionally, a dictionary is returned, that contains the time taken to calculate the prediction.
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
    if isinstance(y_proba, DenseMatrix):
        y_pred = _predict_weighted_per_instance_dense(
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
    y_proba: Matrix,
    k: int,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    r"""
    Predicts the top **k** labels for each instance (row) in a provided matrix of conditional probabilities estimates of labels :math:`\eta` (**y_proba**).
    It is equivalent to calling ``predict_weighted_per_instance(y_proba, k=k, a=None, b=None, return_meta=return_meta)``.

    Args:
        y_proba: A 2D matrix of conditional probabilities for each label.
        k: The number of labels to predict for each instance.
        return_meta: Whether to return metadata. Defaults to False.

    Returns:
        The prediction matrix, with exactly **k** labels in each row, the shape and type of the matrix is the same as **y_proba**. If **return_meta** is True, additionally, a dictionary is returned, that contains the time taken to calculate the prediction.
    """
    return predict_weighted_per_instance(y_proba, k=k, return_meta=return_meta)


def predict_optimizing_macro_recall(
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
    dtype: Optional[DType] = None,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    r"""
    Predicts the top **k** labels for each instance (row)
    in a provided matrix of conditional probabilities estimates of labels :math:`\eta` (**y_proba**).
    The return prediction optimizes macro-averaged recall
    for the population with the given prior probabilities of labels (**priors**).
    It is equivalent to calling ``predict_weighted_per_instance(y_proba, k=k, a=1.0 / (priors + epsilon), return_meta=return_meta)``.

    Args:
        y_proba: A 2D matrix of conditional probabilities for each label.
        k: The number of labels to predict for each instance.
        priors: The prior probabilities for each label.
        epsilon: A small value to avoid division by zero when calculating inverse of priors.
        dtype: The data type for the output matrix, if equal to None, the data type of **y_proba** will be used.
        return_meta: Whether to return metadata.

    Returns:
        The prediction matrix, with exactly **k** labels in each row, the shape and type of the matrix is the same as **y_proba**. If **return_meta** is True, additionally, a dictionary is returned, that contains the time taken to calculate the prediction.
    """
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = 1.0 / (priors + epsilon)
    return predict_weighted_per_instance(
        y_proba, k=k, a=weights, return_meta=return_meta
    )


def _predict_optimizing_macro_balanced_accuracy_dense(
    y_proba: DenseMatrix,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
    dtype: Optional[DType] = None,
) -> Matrix:
    n, m = y_proba.shape
    priors = priors + epsilon
    y_pred = zeros_like(y_proba, dtype=dtype)

    gains = y_proba / priors - (1 - y_proba) / (1 - priors)

    if k > 0:
        # Numpy implementation
        if isinstance(y_proba, np.ndarray):
            top_k = np.argpartition(-gains, k, axis=1)[:, :k]
            y_pred[np.arange(n)[:, None], top_k] = 1

        # Torch implementation
        elif TORCH_AVAILABLE and isinstance(y_proba, torch.Tensor):
            _, top_k = torch.topk(gains, k, axis=1)
            y_pred[torch.arange(n)[:, None], top_k] = 1
    else:
        y_pred[gains >= 0] = 1

    return y_pred


def _predict_optimizing_macro_balanced_accuracy_csr(
    y_proba: csr_matrix,
    k: int,
    priors: np.array,
    epsilon: float = 1e-6,
    dtype: Optional[DType] = None,
) -> csr_matrix:
    n, m = y_proba.shape
    priors = priors + epsilon

    data, indices, indptr = numba_predict_macro_balanced_accuracy_csr(
        *unpack_csr_matrix(y_proba),
        n,
        m,
        k,
        priors,
    )
    return csr_matrix(
        (data, indices, indptr),
        shape=y_proba.shape,
        dtype=y_proba.dtype if dtype is None else dtype,
    )


def predict_optimizing_macro_balanced_accuracy(
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
    dtype: Optional[DType] = None,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    r"""
    Predicts labels for each instance (row)
    in provided matrix of conditional probabilities estimates of labels :math:`\eta` (**y_proba**).
    Such that the prediction at **k** optimizes macro-averaged balanced accuracy
    for the population with the given prior probabilities of labels (**priors**).

    Args:
        y_proba: A 2D matrix of conditional probabilities for each label.
        k: The number of labels to predict for each instance.
        priors: The prior probabilities for each label.
        epsilon: A small value to avoid division by zero when calculating inverse of priors.
        dtype: The data type for the output matrix, if equal to None, the data type of **y_proba** will be used.
        return_meta: Whether to return meta data.

    Returns:
        The prediction matrix, the shape and type of the matrix is the same as **y_proba**. If **return_meta** is True, additionally, a dictionary is returned, that contains the time taken to calculate the prediction.
    """
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    if return_meta:
        meta = {"iters": 1, "time": time()}

    if isinstance(y_proba, DenseMatrix):
        y_pred = _predict_optimizing_macro_balanced_accuracy_dense(
            y_proba, k, priors, epsilon=epsilon, dtype=dtype
        )
    elif isinstance(y_proba, csr_matrix):
        y_pred = _predict_optimizing_macro_balanced_accuracy_csr(
            y_proba, k, priors, epsilon=epsilon, dtype=dtype
        )
    else:
        raise ValueError(
            "y_proba must be either np.ndarray, torch.Tensor, or csr_matrix"
        )

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


def predict_log_weighted_per_instance(
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    epsilon: float = 1e-6,
    dtype: Optional[DType] = None,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    r"""
    Predicts the top **k** labels for each instance (row) in provided matrix of conditional probabilities estimates of labels :math:`\eta` (**y_proba**) according to the log law weighting scheme:

    .. math::
        a = -\log \pi

    where :math:`\pi` (**priors**) is the prior probability of each label, :math:`\beta` (beta) is power parameter and :math:`\epsilon` (**epsilon**) is a small value to avoid domain error.

    It is equivalent to calling ``predict_weighted_per_instance(y_proba, k=k, a=-log(priors + epsilon), return_meta=return_meta)``.

    Args:
        y_proba: A 2D matrix of conditional probabilities for each label.
        k: The number of labels to predict for each instance.
        priors: The prior probabilities for each label.
        epsilon: A small value to avoid domain error.
        dtype: The data type for the output matrix, if equal to None, the data type of **y_proba** will be used.
        return_meta: Whether to return meta data.

    Result
        The prediction matrix, with exactly **k** labels in each row, the shape and type of the matrix is the same as **y_proba**. If **return_meta** is True, additionally, a dictionary is returned, that contains the time taken to calculate the prediction.
    """

    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = -np.log(priors + epsilon)
    return predict_weighted_per_instance(
        y_proba, k=k, a=weights, dtype=dtype, return_meta=return_meta
    )


def predict_power_law_weighted_per_instance(
    y_proba: Matrix,
    k: int,
    priors: DenseMatrix,
    beta: float,
    epsilon: float = 1e-6,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    r"""
    Predicts the top **k** labels for each instance (row) in provided matrix of conditional probabilities estimates of labels :math:`\eta` (**y_proba**) according to the power law weighting scheme:

    .. math::
        a = (\pi + \epsilon)^{-\beta}

    where :math:`\pi` (**priors**) is the prior probability of each label, :math:`\beta` (beta) is power parameter and :math:`\epsilon` (**epsilon**) is a small value to avoid division by zero.

    It is equivalent to calling ``predict_weighted_per_instance(y_proba, k=k, a= (priors + epsilon) ** -beta, return_meta=return_meta)``.

    Args:
        y_proba: A 2D matrix of conditional probabilities for each label.
        k: The number of labels to predict for each instance.
        priors: The prior probabilities for each label.
        beta: The power parameter.
        epsilon: A small value to avoid division by zero when calculating inverse of priors.
        return_meta: Whether to return meta data.

    Result
        The prediction matrix, with exactly **k** labels in each row, the shape and type of the matrix is the same as **y_proba**. If **return_meta** is True, additionally, a dictionary is returned, that contains the time taken to calculate the prediction.
    """
    if priors.shape[0] != y_proba.shape[1]:
        raise ValueError("priors must be of shape (y_proba[1],)")

    weights = (priors + epsilon) ** -beta
    return predict_weighted_per_instance(
        y_proba, k=k, a=weights, return_meta=return_meta
    )


def predict_optimizing_instance_precision(
    y_proba: Matrix,
    k: int,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    if k <= 0:
        raise ValueError("k must be > 0")

    return predict_top_k(y_proba, k=k, return_meta=return_meta)


def predict_optimizing_instance_propensity_scored_precision(
    y_proba: Matrix,
    k: int,
    inverse_propensities: Optional[DenseMatrix] = None,
    propensities: Optional[DenseMatrix] = None,
    return_meta: bool = False,
) -> Union[Matrix, Tuple[Matrix, dict]]:
    if inverse_propensities is not None:
        if inverse_propensities.shape[0] != y_proba.shape[1]:
            raise ValueError("inverse_propensities must be of shape (y_proba[1],)")
    elif propensities is not None:
        if propensities.shape[0] != y_proba.shape[1]:
            raise ValueError("propensities must be of shape (y_proba[1],)")
        propensities[propensities == 0] = 1.0
        inverse_propensities = 1.0 / propensities
    else:
        raise ValueError("either inverse_propensities or propensities must be provided")

    return predict_weighted_per_instance(
        y_proba, k=k, a=inverse_propensities, return_meta=return_meta
    )
