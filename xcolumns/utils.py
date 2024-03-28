import inspect
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .numba_csr_functions import *
from .types import TORCH_AVAILABLE, DenseMatrix, DType


if TORCH_AVAILABLE:
    import torch


########################################################################################
# Logger
########################################################################################


logging.basicConfig()
logger = logging.getLogger("xcolumns")
logger.setLevel(logging.INFO)


def log(msg: str, verbose: bool = True, level: int = logging.INFO):
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
# Dense array utilities
########################################################################################


def zeros_like(a: DenseMatrix, shape=None, dtype: DType = None) -> DenseMatrix:
    if isinstance(a, np.ndarray):
        return np.zeros_like(a, shape=shape, dtype=dtype)
    elif isinstance(a, csr_matrix):
        return np.zeros(
            shape if shape is not None else a.shape,
            dtype=dtype if dtype is not None else a.dtype,
        )
    elif TORCH_AVAILABLE and isinstance(a, torch.Tensor):
        return torch.zeros(
            shape if shape is not None else a.shape,
            dtype=dtype if dtype is not None else a.dtype,
            device=a.device,
        )
    else:
        raise ValueError(f"Unsupported type {type(a)}")


def ones_like(a: DenseMatrix, dtype: DType = None) -> DenseMatrix:
    if isinstance(a, np.ndarray):
        return np.ones_like(a, dtype=dtype)
    elif TORCH_AVAILABLE and isinstance(a, torch.Tensor):
        return torch.ones_like(a, dtype=dtype)
    else:
        raise ValueError(f"Unsupported type {type(a)}")


########################################################################################
# Functions for generating matrices with random prediction at k
########################################################################################


def random_at_k_np(
    shape: Tuple[int, int],
    k: int,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    n, m = shape
    y_pred = np.zeros(shape, dtype=dtype)

    rng = np.random.default_rng(seed)
    labels_range = np.arange(m)
    for i in range(n):
        y_pred[i, rng.choice(labels_range, k, replace=False, shuffle=False)] = 1.0
    return y_pred


def random_at_k_csr(
    shape: Tuple[int, int],
    k: int,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
) -> csr_matrix:
    n, m = shape
    y_pred_data, y_pred_indices, y_pred_indptr = numba_random_at_k(
        n, m, k, seed=seed, dtype=dtype
    )
    return construct_csr_matrix(
        y_pred_data.astype(dtype),
        y_pred_indices,
        y_pred_indptr,
        dtype=dtype,
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


########################################################################################
# Function signature utilities
########################################################################################


def add_kwargs_to_signature(
    func: Callable, func_with_kwargs: Callable, skip: Optional[List] = None
) -> Callable:
    if skip is None:
        skip = []

    sig_with_kwargs = inspect.signature(func_with_kwargs)
    sig_new = inspect.signature(func)
    func.__signature__ = sig_new.replace(
        parameters=[
            p
            for p in sig_new.parameters.values()
            if p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        + [
            p
            for p in sig_with_kwargs.parameters.values()
            if p.default != inspect.Parameter.empty and p.name not in skip
        ]
    )

    return func
