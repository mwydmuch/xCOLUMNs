import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

from .default_types import FLOAT_TYPE, INT_TYPE


@njit
def numba_first_k(n: int, k: int):
    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=INT_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=INT_TYPE)
    for i in range(n):
        y_pred_indices[(i * k) : ((i + 1) * k)] = np.arange(0, k, 1, FLOAT_TYPE)
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k
    return y_pred_data, y_pred_indices, y_pred_indptr


@njit
def numba_random_at_k_from(
    indices: np.ndarray, indptr: np.ndarray, n: int, m: int, k: int, seed: int = None
):
    """
    Selects k random labels for each instance.
    """
    if seed is not None:
        np.random.seed(seed)

    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=INT_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=INT_TYPE)
    for i in range(n):
        row_indices = indices[indptr[i] : indptr[i + 1]]
        if row_indices.size >= k:
            y_pred_indices[i * k : (i + 1) * k] = np.random.choice(
                row_indices, k, replace=False
            )
        else:
            y_pred_indices[i * k : (i + 1) * k] = np.random.choice(
                m, k, replace=False
            )
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit
def numba_random_at_k(n: int, m: int, k: int, seed: int = None):
    """
    Selects k random labels for each instance.
    """
    if seed is not None:
        np.random.seed(seed)

    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=INT_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=INT_TYPE)
    for i in range(n):
        y_pred_indices[i * k : (i + 1) * k] = np.random.choice(
            m, k, replace=False
        )
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit
def numba_sparse_vec_mul_vec(
    a_data: np.ndarray, a_indices: np.ndarray, b_data: np.ndarray, b_indices: np.ndarray
):
    """
    Performs a fast multiplication of sparse vectors a and b.
    Gives the same y_pred as a.multiply(b) where a and b are sparse vectors.
    Requires a and b to have sorted indices (in ascending order).
    """
    i = j = k = 0
    new_data_size = min(a_data.size, b_data.size)
    new_data = np.zeros(new_data_size, dtype=FLOAT_TYPE)
    new_indices = np.zeros(new_data_size, dtype=INT_TYPE)
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


@njit
def numba_sparse_vec_mul_ones_minus_vec(
    a_data: np.ndarray, a_indices: np.ndarray, b_data: np.ndarray, b_indices: np.ndarray
):
    """
    Performs a fast multiplication of a sparse vector a
    with a dense vector of ones minus other sparse vector b.
    Gives the same y_pred as a.multiply(ones - b) where a and b are sparse vectors.
    Requires a and b to have sorted indices (in ascending order).
    """
    i = j = k = 0
    new_data_size = a_data.size + b_data.size
    new_data = np.zeros(new_data_size, dtype=FLOAT_TYPE)
    new_indices = np.zeros(new_data_size, dtype=INT_TYPE)
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


@njit
def numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(
    a_data, a_indices, a_indptr, b_data, b_indices, b_indptr, n, m
):
    """
    Performs a fast multiplication of a sparse matrix a
    with a dense matrix of ones minus other sparse matrix b and then sums the rows (axis=0).
    Gives the same y_pred as a.multiply(ones - b) where a and b are sparse matrices.
    Requires a and b to have sorted indices (in ascending order).
    """
    y_pred = np.zeros(m, dtype=FLOAT_TYPE)
    for i in range(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )
        y_pred[indices] += data

    return y_pred


@njit
def numba_calculate_prod_1_sparse_mat_mul_ones_minus_mat(
    a_data, a_indices, a_indptr, b_data, b_indices, b_indptr, n, m
):
    """
    Performs a fast multiplication of a sparse matrix a
    with a dense matrix of ones minus other sparse matrix b and then sums the rows (axis=0).
    Gives the same y_pred as a.multiply(ones - b) where a and b are sparse matrices.
    Requires a and b to have sorted indices (in ascending order).
    """
    y_pred = np.ones(m, dtype=FLOAT_TYPE)
    for i in range(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )
        y_pred[indices] *= data

    return y_pred


@njit
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


@njit
def numba_weighted_per_instance(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    weights: np.ndarray,
    n: int,
    m: int,
    k: int,
):
    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=INT_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=INT_TYPE)

    # This can be done in parallel, but Numba parallelism seems to not work well here
    for i in range(n):
        row_data = data[indptr[i] : indptr[i + 1]]
        row_indices = indices[indptr[i] : indptr[i + 1]]
        row_weights = weights[row_indices].reshape(-1) * row_data
        top_k = numba_argtopk(row_weights, row_indices, k)
        y_pred_indices[i * k : i * k + len(top_k)] = top_k
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit
def numba_macro_balanced_accuracy(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    marginals: np.ndarray,
    n: int,
    m: int,
    k: int,
):
    """
    Predicts k labels for each instance according to the optimal strategy for macro-balanced accuracy.
    """
    y_pred_data = np.ones(n * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(n * k, dtype=INT_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=INT_TYPE)

    for i in range(n):
        row_data = data[indptr[i] : indptr[i + 1]]
        row_indices = indices[indptr[i] : indptr[i + 1]]
        row_marginals = marginals[row_indices].reshape(-1)
        row_gains = row_data / row_marginals - (1 - row_data) / (1 - row_marginals)
        top_k = numba_argtopk(row_gains, row_indices, k)
        y_pred_indices[i * k : i * k + len(top_k)] = top_k
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr