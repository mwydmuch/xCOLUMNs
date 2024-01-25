import random
from typing import Optional, Tuple

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

from .default_types import *


@njit(cache=True)
def numba_first_k(n: int, k: int):
    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=IND_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=IND_TYPE)
    first_k = np.arange(k, dtype=IND_TYPE)
    for i in range(n):
        y_pred_indices[i * k : (i + 1) * k] = first_k
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k
    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=True)
def numba_random_at_k_from(
    indices: np.ndarray, indptr: np.ndarray, n: int, m: int, k: int, seed: int = None
):
    """
    Selects k random labels for each instance.
    """
    # rng = np.random.default_rng(seed) # Numba cannot use new random generator
    if seed is not None:
        np.random.seed(seed)

    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=IND_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=IND_TYPE)
    labels_range = np.arange(m, dtype=IND_TYPE)
    for i in range(n):
        row_indices = indices[indptr[i] : indptr[i + 1]]
        if row_indices.size >= k:
            # y_pred_indices[i * k : (i + 1) * k] = np.random.choice(
            #     row_indices, k, replace=False
            # )
            y_pred_indices[i * k : (i + 1) * k] = numba_fast_random_choice(
                row_indices, k
            )
        else:
            # y_pred_indices[i * k : (i + 1) * k] = np.random.choice(
            #     labels_range, k, replace=False
            # )
            y_pred_indices[i * k : (i + 1) * k] = numba_fast_random_choice(
                labels_range, k
            )
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=True)
def numba_fast_random_choice(array, k=-1):
    """
    Selects k random elements from array.
    """
    n = array.size
    if k < 0:
        k = array.size
    index = np.arange(n, dtype=IND_TYPE)
    for i in range(k):
        j = random.randint(i, n - 1)
        index[i], index[j] = index[j], index[i]
    return array[index[:k]]


@njit(cache=True)
def numba_random_at_k(n: int, m: int, k: int, seed: int = None):
    """
    Selects k random labels for each instance.
    """
    # rng = np.random.default_rng(seed) # Numba cannot use new random generator
    if seed is not None:
        # np.random.seed(seed) np.random.choice seems to be quite slow here
        random.seed(seed)

    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=IND_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=IND_TYPE)
    labels_range = np.arange(m, dtype=IND_TYPE)
    for i in range(n):
        # y_pred_indices[i * k : (i + 1) * k] = np.random.choice(
        #     labels_range, k, replace=False
        # )
        y_pred_indices[i * k : (i + 1) * k] = numba_fast_random_choice(labels_range, k)
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=True)
def numba_sparse_vec_mul_vec(
    a_data: np.ndarray, a_indices: np.ndarray, b_data: np.ndarray, b_indices: np.ndarray
):
    """
    Performs a fast multiplication of sparse vectors a and b.
    Gives the same result as a.multiply(b) where a and b are sparse vectors.
    Requires a and b to have sorted indices (in ascending order).
    """
    i = j = k = 0
    new_data_size = min(a_data.size, b_data.size)
    new_data = np.zeros(new_data_size, dtype=FLOAT_TYPE)
    new_indices = np.zeros(new_data_size, dtype=IND_TYPE)
    while i < a_indices.size and j < b_indices.size:
        # print(i, j, k, a_indices[i], b_indices[j], a_indices.size, b_indices.size)
        if a_indices[i] < b_indices[j]:
            i += 1
        elif a_indices[i] == b_indices[j]:
            new_data[k] = a_data[i] * b_data[j]
            new_indices[k] = a_indices[i]
            k += 1
            i += 1
            j += 1
        else:
            j += 1
    return new_data[:k], new_indices[:k]


@njit(cache=True)
def numba_calculate_sum_0_sparse_mat_mul_mat(
    a_data: np.ndarray,
    a_indices: np.ndarray,
    a_indptr: np.ndarray,
    b_data: np.ndarray,
    b_indices: np.ndarray,
    b_indptr: np.ndarray,
    n: int,
    m: int,
):
    """
    Performs a fast multiplication of sparse matricies a and b.
    Gives the same result as a.multiply(b).sum(axis=0) where a and b are sparse vectors.
    Requires a and b to have sorted indices (in ascending order).
    """
    result = np.zeros(m, dtype=FLOAT_TYPE)
    for i in range(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        data, indices = numba_sparse_vec_mul_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )
        result[indices] += data

    return result


@njit(cache=True)
def numba_sparse_vec_mul_ones_minus_vec(
    a_data: np.ndarray, a_indices: np.ndarray, b_data: np.ndarray, b_indices: np.ndarray
):
    """
    Performs a fast multiplication of a sparse vector a
    with a dense vector of ones minus other sparse vector b.
    Gives the same result as a.multiply(ones - b) where a and b are sparse vectors but makes it more efficient.
    Requires a and b to have sorted indices (in ascending order).
    """
    i = j = k = 0
    new_data_size = a_data.size + b_data.size
    new_data = np.zeros(new_data_size, dtype=FLOAT_TYPE)
    new_indices = np.zeros(new_data_size, dtype=IND_TYPE)
    while i < a_indices.size:
        # print(i, j, k, a_indices[i], b_indices[j], a_indices.size, b_indices.size)
        if j >= b_indices.size or a_indices[i] < b_indices[j]:
            new_data[k] = a_data[i]
            new_indices[k] = a_indices[i]
            k += 1
            i += 1
        elif a_indices[i] == b_indices[j]:
            new_data[k] = a_data[i] * (1 - b_data[j])
            new_indices[k] = a_indices[i]
            k += 1
            i += 1
            j += 1
        else:
            j += 1
    return new_data[:k], new_indices[:k]


@njit(cache=True)
def numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(
    a_data: np.ndarray,
    a_indices: np.ndarray,
    a_indptr: np.ndarray,
    b_data: np.ndarray,
    b_indices: np.ndarray,
    b_indptr: np.ndarray,
    n: int,
    m: int,
):
    """
    Performs a fast multiplication of a sparse matrix a
    with a dense matrix of ones minus other sparse matrix b and then sums the rows (axis=0).
    Gives the same result as a.multiply(ones - b).sum(axis=0) where a and b are sparse matrices but makes it more efficient.
    Requires a and b to have sorted indices (in ascending order).
    """
    result = np.zeros(m, dtype=FLOAT_TYPE)
    for i in range(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )
        result[indices] += data

    return result


# TODO: fix docstirng
@njit(cache=True)
def numba_calculate_prod_0_sparse_mat_mul_ones_minus_mat(
    a_data, a_indices, a_indptr, b_data, b_indices, b_indptr, n, m
):
    """
    Performs a fast multiplication of a sparse matrix a
    with a dense matrix of ones minus other sparse matrix b and then sums the rows (axis=0).
    Gives the same result as a.multiply(ones - b).prod(axis=0) where a and b are sparse matrices but makes it more efficient.
    Requires a and b to have sorted indices (in ascending order).
    """
    result = np.ones(m, dtype=FLOAT_TYPE)
    for i in range(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )
        result[indices] *= data

    return result


@njit(cache=True)
def numba_argtopk(data, indices, k):
    """
    Returns the indices of the top k elements
    """
    if data.size > k:
        top_k = indices[np.argpartition(-data, k)[:k]]
        top_k.sort()
        return top_k
    else:
        return indices


@njit(cache=True)
def numba_resize(arr, new_size):
    new_arr = np.zeros(new_size, arr.dtype)
    new_arr[: arr.size] = arr
    return new_arr


# @njit(cache=True)
# def numba_predict_weighted_per_instance(
#     data: np.ndarray,
#     indices: np.ndarray,
#     indptr: np.ndarray,
#     #shape: Tuple[int, int],
#     n: int,
#     m: int,
#     k: int,
#     a: Optional[np.ndarray] = None,
#     b: Optional[np.ndarray] = None,
# ):
#     y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
#     y_pred_indices = np.zeros(n * k, dtype=IND_TYPE)
#     y_pred_indptr = np.zeros(n + 1, dtype=IND_TYPE)

#     # This can be done in parallel, but Numba parallelism seems to not work well here
#     for i in range(n):
#         row_data = data[indptr[i] : indptr[i + 1]]
#         row_indices = indices[indptr[i] : indptr[i + 1]]

#         row_gains = row_data
#         if a is not None:
#             row_gains = row_gains * a[row_indices].reshape(-1)
#         if b is not None:
#             row_gains = row_gains + b[row_indices].reshape(-1)

#         top_k = numba_argtopk(row_gains, row_indices, k)
#         y_pred_indices[i * k : i * k + len(top_k)] = top_k
#         y_pred_indptr[i + 1] = y_pred_indptr[i] + k

#     return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=True)
def numba_predict_weighted_per_instance(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    # shape: Tuple[int, int],
    n: int,
    m: int,
    k: int = 0,
    t: float = 0,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
):
    # This can be done in parallel, but Numba parallelism seems to not work well here
    y_pred_data = np.ones(n * max(1, k), dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * max(1, k), dtype=IND_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=IND_TYPE)

    for i in range(n):
        row_data = data[indptr[i] : indptr[i + 1]]
        row_indices = indices[indptr[i] : indptr[i + 1]]

        row_gains = row_data
        if a is not None:
            row_gains = row_gains * a[row_indices].reshape(-1)
        if b is not None:
            row_gains = row_gains + b[row_indices].reshape(-1)

        if k > 0:
            top_k = numba_argtopk(row_gains, row_indices, k)
            y_pred_indices[i * k : i * k + len(top_k)] = top_k
            y_pred_indptr[i + 1] = y_pred_indptr[i] + k
        else:
            selected_indices = row_indices[row_gains >= t]

            if y_pred_indptr[i] + len(selected_indices) >= len(y_pred_indices):
                y_pred_data = numba_resize(y_pred_data, len(y_pred_data) * 2)
                y_pred_indices = numba_resize(y_pred_indices, len(y_pred_indices) * 2)

            y_pred_indices[
                y_pred_indptr[i] : y_pred_indptr[i] + len(selected_indices)
            ] = selected_indices
            y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=True)
def numba_predict_macro_balanced_accuracy(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    n: int,
    m: int,
    k: int,
    marginals: np.ndarray,
):
    """
    Predicts k labels for each instance according to the optimal strategy for macro-balanced accuracy.
    """
    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=IND_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=IND_TYPE)

    for i in range(n):
        row_data = data[indptr[i] : indptr[i + 1]]
        row_indices = indices[indptr[i] : indptr[i + 1]]
        row_marginals = marginals[row_indices].reshape(-1)
        row_gains = row_data / row_marginals - (1 - row_data) / (1 - row_marginals)
        top_k = numba_argtopk(row_gains, row_indices, k)
        y_pred_indices[i * k : i * k + len(top_k)] = top_k
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr
