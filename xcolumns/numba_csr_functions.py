import os
import random
from typing import Optional, Tuple

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.sparse import csr_matrix

from .types import CSRMatrixAsTuple, DefaultDataDType, DefaultIndDType


def str_bool_true(value: str) -> bool:
    """
    Checks if string represents a boolean true value.
    """
    return value.lower() in ("true", "1")


NUMBA_CACHE = str_bool_true(os.environ.get("XCOLUMNS_NUMBA_CACHE", "0"))
NUMBA_PARALLEL = str_bool_true(os.environ.get("XCOLUMNS_NUMBA_PARALLEL", "0"))
NUMBA_THREADS = os.environ.get("NUMBA_NUM_THREADS", 1)


@njit(cache=NUMBA_CACHE)
def numba_first_k(
    n: int, k: int, dtype: np.dtype = DefaultDataDType
) -> CSRMatrixAsTuple:
    """
    Selects the first k labels indexes ([0, 1, ..., k]) for each instance.
    """
    y_pred_data = np.ones(n * k, dtype=DefaultDataDType)
    y_pred_indices = np.zeros(n * k, dtype=DefaultIndDType)
    y_pred_indptr = np.zeros(n + 1, dtype=DefaultIndDType)
    first_k = np.arange(k, dtype=DefaultIndDType)
    for i in range(n):
        y_pred_indices[i * k : (i + 1) * k] = first_k
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k
    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=NUMBA_CACHE)
def numba_random_at_k_from(
    y_proba_indices: np.ndarray,
    y_proba_indptr: np.ndarray,
    n: int,
    m: int,
    k: int,
    seed: Optional[int] = None,
) -> CSRMatrixAsTuple:
    """
    Selects k random labels for each instance.
    """
    # rng = np.random.default_rng(seed) # Numba cannot use new random generator
    if seed is not None:
        np.random.seed(seed)

    y_pred_data = np.ones(n * k, dtype=DefaultDataDType)
    y_pred_indices = np.zeros(n * k, dtype=y_proba_indices)
    y_pred_indptr = np.zeros(n + 1, dtype=y_proba_indptr)
    labels_range = np.arange(m, dtype=y_proba_indices)
    for i in range(n):
        row_indices = y_proba_indices[y_proba_indptr[i] : y_proba_indptr[i + 1]]
        if row_indices.size >= k:
            # This is faster then above np.random.choice(..., replace=False)
            y_pred_indices[i * k : (i + 1) * k] = numba_fast_random_choice(
                row_indices, k
            )
        else:
            y_pred_indices[i * k : (i + 1) * k] = numba_fast_random_choice(
                labels_range, k
            )
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=NUMBA_CACHE)
def numba_fast_random_choice(array, k=-1) -> np.ndarray:
    """
    Selects k random elements from array.
    """
    n = array.size
    if k < 0:
        k = array.size
    index = np.arange(n, dtype=DefaultIndDType)
    for i in range(k):
        j = random.randint(i, n - 1)
        index[i], index[j] = index[j], index[i]
    return array[index[:k]]


@njit(cache=NUMBA_CACHE)
def numba_random_at_k(
    n: int, m: int, k: int, seed: Optional[int] = None, dtype: Optional[np.dtype] = None
) -> CSRMatrixAsTuple:
    """
    Selects k random labels for each instance.
    """
    # rng = np.random.default_rng(seed) # Numba cannot use new random generator
    if seed is not None:
        # np.random.seed(seed) np.random.choice seems to be quite slow here
        random.seed(seed)

    y_pred_data = np.ones(n * k, dtype=DefaultDataDType)
    y_pred_indices = np.zeros(n * k, dtype=DefaultIndDType)
    y_pred_indptr = np.zeros(n + 1, dtype=DefaultIndDType)
    labels_range = np.arange(m, dtype=DefaultIndDType)
    for i in range(n):
        y_pred_indices[i * k : (i + 1) * k] = numba_fast_random_choice(labels_range, k)
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=NUMBA_CACHE)
def numba_csr_vec_mul_vec(
    a_data: np.ndarray, a_indices: np.ndarray, b_data: np.ndarray, b_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a fast multiplication of sparse vectors a and b.
    Gives the same result as a.multiply(b) where a and b are sparse vectors.
    Requires a and b to have sorted y_proba_indices (in ascending order).
    """
    i = j = k = 0
    new_data_size = min(a_data.size, b_data.size)
    new_data = np.zeros(new_data_size, dtype=a_data.dtype)
    new_indices = np.zeros(new_data_size, dtype=a_indices.dtype)
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


@njit(cache=NUMBA_CACHE, parallel=NUMBA_PARALLEL)
def numba_calculate_sum_csr_mat_mul_mat(
    a_data: np.ndarray,
    a_indices: np.ndarray,
    a_indptr: np.ndarray,
    b_data: np.ndarray,
    b_indices: np.ndarray,
    b_indptr: np.ndarray,
    n: int,
    m: int,
    dtype: np.dtype,
    axis: int = 0,
) -> np.ndarray:
    """
    Performs a fast multiplication of sparse matrices a and b.
    Gives the same result as a.multiply(b).sum(axis=axis) where a and b are sparse vectors.
    Requires a and b to have sorted y_proba_indices (in ascending order).
    """
    if axis == 0:
        result = np.zeros(m, dtype=dtype)
    else:
        result = np.zeros(n, dtype=dtype)

    for i in prange(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        y_proba_data, y_proba_indices = numba_csr_vec_mul_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )

        if axis == 0:
            result[y_proba_indices] += y_proba_data
        else:
            result[i] = np.sum(y_proba_data)

    return result


@njit(cache=NUMBA_CACHE)
def numba_csr_vec_mul_ones_minus_vec(
    a_data: np.ndarray, a_indices: np.ndarray, b_data: np.ndarray, b_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a fast multiplication of a sparse vector a
    with a dense vector of ones minus other sparse vector b.
    Gives the same result as a.multiply(ones - b) where a and b are sparse vectors but makes it more efficient.
    Requires a and b to have sorted y_proba_indices (in ascending order).
    """
    i = j = k = 0
    new_data_size = a_data.size + b_data.size
    new_data = np.zeros(new_data_size, dtype=a_data.dtype)
    new_indices = np.zeros(new_data_size, dtype=a_indices.dtype)
    while i < a_indices.size:
        if j >= b_indices.size or a_indices[i] < b_indices[j]:
            new_data[k] = a_data[i]
            new_indices[k] = a_indices[i]
            k += 1
            i += 1
        elif a_indices[i] == b_indices[j]:
            new_data[k] = a_data[i] * (1.0 - b_data[j])
            new_indices[k] = a_indices[i]
            k += 1
            i += 1
            j += 1
        else:
            j += 1
    return new_data[:k], new_indices[:k]


@njit(cache=NUMBA_CACHE, parallel=NUMBA_PARALLEL)
def numba_calculate_sum_csr_mat_mul_ones_minus_mat(
    a_data: np.ndarray,
    a_indices: np.ndarray,
    a_indptr: np.ndarray,
    b_data: np.ndarray,
    b_indices: np.ndarray,
    b_indptr: np.ndarray,
    n: int,
    m: int,
    dtype: np.dtype,
    axis: int = 0,
) -> np.ndarray:
    """
    Performs a fast multiplication of a sparse matrix a
    with a dense matrix of ones minus other sparse matrix b and then sums the rows (axis=0) or columns (axis=1).
    Gives the same result as a.multiply(ones - b).sum(axis=axis) where a and b are sparse matrices but makes it more efficient.
    Requires a and b to have sorted y_proba_indices (in ascending order).
    """
    if axis == 0:
        result = np.zeros(m, dtype=dtype)
    elif axis == 1:
        result = np.zeros(n, dtype=dtype)
    else:
        raise ValueError("axis must be 0 or 1")

    for i in prange(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        data, indices = numba_csr_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )

        if axis == 0:
            result[indices] += data
        else:
            result[i] = np.sum(data)

    return result


@njit(cache=NUMBA_CACHE)
def numba_calculate_prod_ones_minus_csr_mat_mul_mat(
    a_data: np.ndarray,
    a_indices: np.ndarray,
    a_indptr: np.ndarray,
    b_data: np.ndarray,
    b_indices: np.ndarray,
    b_indptr: np.ndarray,
    n: int,
    m: int,
    dtype: np.dtype,
    axis: int = 0,
    use_log: bool = False,
) -> np.ndarray:
    """
    Performs a fast substraction from dense matrix ones a sparse matrix a mutiply with other sparse matrix b
    and then multiply the rows (axis=0) or columns (axis=1).
    Gives the same result as (ones - a.multiply(b)).prod(axis=axis) where a and b are sparse matrices but makes it more efficient.
    Requires a and b to have sorted y_proba_indices (in ascending order).
    """
    size = 0
    if axis == 0:
        size = m
    elif axis == 1:
        size = n
    else:
        raise ValueError("axis must be 0 or 1")

    if use_log:
        result = np.zeros(size, dtype=dtype)
    else:
        result = np.ones(size, dtype=dtype)

    for i in prange(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        data, indices = numba_csr_vec_mul_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )

        if use_log:
            data = np.log(1.0 - data)
            if axis == 0:
                result[indices] += data
            else:
                result[i] = np.sum(data)
        else:
            data = 1.0 - data
            if axis == 0:
                result[indices] *= data
            else:
                result[i] = np.prod(data)

    if use_log:
        result = np.exp(result)

    return result


@njit(cache=NUMBA_CACHE)
def numba_calculate_prod_csr_mat_mul_ones_minus_mat(
    a_data: np.ndarray,
    a_indices: np.ndarray,
    a_indptr: np.ndarray,
    b_data: np.ndarray,
    b_indices: np.ndarray,
    b_indptr: np.ndarray,
    n: int,
    m: int,
    dtype: np.dtype,
    axis: int = 0,
    use_log: bool = False,
) -> np.ndarray:
    """
    Performs a fast multiplication of a sparse matrix a
    with a dense matrix of ones minus other sparse matrix b and then multiply the rows (axis=0) or columns (axis=1).
    Gives the same result as a.multiply(ones - b).prod(axis=axis) where a and b are sparse matrices but makes it more efficient.
    Requires a and b to have sorted y_proba_indices (in ascending order).
    """
    size = 0
    if axis == 0:
        size = m
    elif axis == 1:
        size = n
    else:
        raise ValueError("axis must be 0 or 1")

    if use_log:
        result = np.zeros(size, dtype=dtype)
    else:
        result = np.ones(size, dtype=dtype)

    for i in prange(n):
        a_start, a_end = a_indptr[i], a_indptr[i + 1]
        b_start, b_end = b_indptr[i], b_indptr[i + 1]

        data, indices = numba_csr_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
        )
        if use_log:
            data = np.log(data)
            if axis == 0:
                result[indices] += data
            else:
                result[i] = np.sum(data)
        else:
            if axis == 0:
                result[indices] *= data
            else:
                result[i] = np.prod(data)

    if use_log:
        result = np.exp(result)

    return result


@njit(cache=NUMBA_CACHE)
def numba_sub_from_unnormalized_confusion_matrix_csr(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
    true_data: np.ndarray,
    true_indices: np.ndarray,
    pred_data: np.ndarray,
    pred_indices: np.ndarray,
    skip_tn=False,
) -> None:
    """
    Updates the confusion matrix by substracting the values based on the true and predicted labels.
    """
    tp_data, tp_indices = numba_csr_vec_mul_vec(
        pred_data, pred_indices, true_data, true_indices
    )
    tp[tp_indices] -= tp_data
    ft_data, ft_indices = numba_csr_vec_mul_ones_minus_vec(
        pred_data, pred_indices, true_data, true_indices
    )
    fp[ft_indices] -= ft_data
    fn_data, fn_indices = numba_csr_vec_mul_ones_minus_vec(
        true_data, true_indices, pred_data, pred_indices
    )
    fn[fn_indices] -= fn_data

    if not skip_tn:
        tn -= 1
        tn[tp_indices] += tp_data
        tn[ft_indices] += ft_data
        tn[fn_indices] += fn_data


@njit(cache=NUMBA_CACHE)
def numba_add_to_unnormalized_confusion_matrix_csr(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
    true_data: np.ndarray,
    true_indices: np.ndarray,
    pred_data: np.ndarray,
    pred_indices: np.ndarray,
    skip_tn=False,
) -> None:
    """
    Updates the confusion matrix by adding the values based on the true and predicted labels.
    """
    tp_data, tp_indices = numba_csr_vec_mul_vec(
        pred_data, pred_indices, true_data, true_indices
    )
    tp[tp_indices] += tp_data
    ft_data, ft_indices = numba_csr_vec_mul_ones_minus_vec(
        pred_data, pred_indices, true_data, true_indices
    )
    fp[ft_indices] += ft_data
    fn_data, fn_indices = numba_csr_vec_mul_ones_minus_vec(
        true_data, true_indices, pred_data, pred_indices
    )
    fn[fn_indices] += fn_data

    if not skip_tn:
        tn += 1
        tn[tp_indices] -= tp_data
        tn[ft_indices] -= ft_data
        tn[fn_indices] -= fn_data


@njit(cache=NUMBA_CACHE)
def numba_argtopk_csr(y_proba_data, y_proba_indices, k, sort=True) -> np.ndarray:
    """
    Returns the y_proba_indices of the top k elements.
    """
    if y_proba_data.size > k:
        topk_arg = y_proba_indices[np.argpartition(-y_proba_data, k)[:k]]
        if sort:
            topk_arg.sort()
        return topk_arg
    else:
        return y_proba_indices


@njit(cache=NUMBA_CACHE)
def numba_topk_csr(y_proba_data, y_proba_indices, k, sort=True) -> np.ndarray:
    """
    Returns the y_proba_indices of the top k elements.
    """
    if y_proba_data.size > k:
        data_topk_arg = np.argpartition(-y_proba_data, k)[:k]
        topk_val = y_proba_data[data_topk_arg]
        topk_arg = y_proba_indices[data_topk_arg]
        if sort:
            sorted_order = topk_arg.argsort()
            topk_arg = topk_arg[sorted_order]
            topk_val = topk_val[sorted_order]
        return topk_arg, topk_val
    else:
        return y_proba_indices, y_proba_data


@njit(cache=NUMBA_CACHE)
def numba_resize(arr, new_size, fill) -> np.ndarray:
    """
    Resizes the array to new_size and fills the rest with fill.
    """
    new_arr = np.zeros(new_size, dtype=arr.dtype)
    new_arr[: arr.size] = arr
    if fill != 0:
        new_arr[arr.size :] = fill
    return new_arr


@njit(cache=NUMBA_CACHE)
def numba_set_gains_csr(
    y_pred_data: np.ndarray,
    y_pred_indices: np.ndarray,
    y_pred_indptr: np.ndarray,
    gains_data: np.ndarray,
    gains_indices: np.ndarray,
    i: int,
    k: int,
    th: float,
    is_insert: bool = True,
) -> CSRMatrixAsTuple:
    """
    Sets top k gains or gains > th for the i-th instance in the y_pred csr matrix.
    """
    if k > 0:
        new_y_pred_i_indices = numba_argtopk_csr(gains_data, gains_indices, k)
    else:
        new_y_pred_i_indices = gains_indices[gains_data >= th]

    y_pred_i_start = y_pred_indptr[i]
    y_pred_i_end = y_pred_indptr[i + 1]

    # Prediction match previous size, just update y_proba_indices
    if y_pred_i_start + new_y_pred_i_indices.size == y_pred_i_end:
        y_pred_indices[y_pred_i_start:y_pred_i_end] = new_y_pred_i_indices
    else:
        new_y_pred_i_end = y_pred_i_start + new_y_pred_i_indices.size
        new_y_pred_last_indptr = y_pred_indptr[-1] + new_y_pred_i_end - y_pred_i_end

        # Resize if needed
        if y_pred_indices.size <= new_y_pred_last_indptr:
            new_y_pred_size = max(y_pred_indices.size * 2, new_y_pred_last_indptr)
            y_pred_indices = numba_resize(y_pred_indices, new_y_pred_size, 0)
            y_pred_data = numba_resize(y_pred_data, new_y_pred_size, 1.0)

        # Move rest of the y_pred_data
        if is_insert:
            y_pred_indices[new_y_pred_i_end:new_y_pred_last_indptr] = y_pred_indices[
                y_pred_i_end : y_pred_indptr[-1]
            ]
            y_pred_indptr[i + 2 :] += new_y_pred_i_end - y_pred_i_end

        # Assign new y_pred_indices
        y_pred_indices[y_pred_i_start:new_y_pred_i_end] = new_y_pred_i_indices
        y_pred_indptr[i + 1] = new_y_pred_i_end

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=NUMBA_CACHE)
def numba_predict_weighted_per_instance_csr_step(
    y_pred_data: np.ndarray,
    y_pred_indices: np.ndarray,
    y_pred_indptr: np.ndarray,
    y_proba_data: np.ndarray,
    y_proba_indices: np.ndarray,
    y_proba_indptr: np.ndarray,
    i: int,
    k: int,
    th: float,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    is_insert=True,
) -> CSRMatrixAsTuple:
    gains_data = y_proba_data[y_proba_indptr[i] : y_proba_indptr[i + 1]]
    gains_indices = y_proba_indices[y_proba_indptr[i] : y_proba_indptr[i + 1]]

    if a is not None:
        gains_data = gains_data * a[gains_indices].reshape(-1)
    if b is not None:
        gains_data = gains_data + b[gains_indices].reshape(-1)

    return numba_set_gains_csr(
        y_pred_data,
        y_pred_indices,
        y_pred_indptr,
        gains_data,
        gains_indices,
        i,
        k,
        th,
        is_insert=is_insert,
    )


@njit(cache=NUMBA_CACHE, parallel=NUMBA_PARALLEL)
def numba_predict_weighted_per_instance_csr(
    y_proba_data: np.ndarray,
    y_proba_indices: np.ndarray,
    y_proba_indptr: np.ndarray,
    # shape: Tuple[int, int],
    k: int = 0,
    th: float = 0,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    keep_scores: bool = False,
) -> CSRMatrixAsTuple:
    n = y_proba_indptr.size - 1
    initial_row_size = k if k > 0 else 10
    y_pred_data = np.ones(n * initial_row_size, dtype=y_proba_data.dtype)
    y_pred_indices = np.zeros(n * initial_row_size, dtype=y_proba_indices.dtype)
    y_pred_indptr = np.arange(n + 1, dtype=y_proba_indptr.dtype) * initial_row_size

    if k > 0:
        for i in prange(n):
            gains_data = y_proba_data[y_proba_indptr[i] : y_proba_indptr[i + 1]]
            gains_indices = y_proba_indices[y_proba_indptr[i] : y_proba_indptr[i + 1]]

            if a is not None:
                gains_data = gains_data * a[gains_indices].reshape(-1)
            if b is not None:
                gains_data = gains_data + b[gains_indices].reshape(-1)

            if keep_scores:
                new_y_pred_i_indices, new_y_pred_i_data = numba_topk_csr(
                    gains_data, gains_indices, k
                )
                y_pred_indices[
                    y_pred_indptr[i] : y_pred_indptr[i]
                    + len(new_y_pred_i_indices)
                    # Instead of y_pred_indptr[i] : y_pred_indptr[i + 1], becasue len(new_y_pred_i_indices) can be less than k
                ] = new_y_pred_i_indices
                y_pred_data[
                    y_pred_indptr[i] : y_pred_indptr[i] + len(new_y_pred_i_indices)
                ] = new_y_pred_i_data
            else:
                new_y_pred_i_indices = numba_argtopk_csr(gains_data, gains_indices, k)
                y_pred_indices[
                    y_pred_indptr[i] : y_pred_indptr[i] + len(new_y_pred_i_indices)
                ] = new_y_pred_i_indices

    else:
        for i in range(n):
            (
                y_pred_data,
                y_pred_indices,
                y_pred_indptr,
            ) = numba_predict_weighted_per_instance_csr_step(
                y_pred_data,
                y_pred_indices,
                y_pred_indptr,
                y_proba_data,
                y_proba_indices,
                y_proba_indptr,
                i,
                k,
                th,
                a,
                b,
                is_insert=False,
            )

        y_pred_indices = y_pred_indices[: y_pred_indptr[-1]]
        y_pred_data = y_pred_data[: y_pred_indptr[-1]]

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit(cache=NUMBA_CACHE)
def numba_predict_macro_balanced_accuracy_csr(
    y_proba_data: np.ndarray,
    y_proba_indices: np.ndarray,
    y_proba_indptr: np.ndarray,
    n: int,
    m: int,
    k: int,
    marginals: np.ndarray,
) -> CSRMatrixAsTuple:
    """
    Predicts k labels for each instance according to the optimal strategy for macro-balanced accuracy.
    """
    y_pred_data = np.ones(n * k, dtype=y_proba_data.dtype)
    y_pred_indices = np.zeros(n * k, dtype=y_proba_indices.dtype)
    y_pred_indptr = np.zeros(n + 1, dtype=y_proba_indptr.dtype)

    for i in range(n):
        row_data = y_proba_data[y_proba_indptr[i] : y_proba_indptr[i + 1]]
        row_indices = y_proba_indices[y_proba_indptr[i] : y_proba_indptr[i + 1]]
        row_marginals = marginals[row_indices].reshape(-1)
        row_gains = row_data / row_marginals - (1 - row_data) / (1 - row_marginals)
        top_k = numba_argtopk_csr(row_gains, row_indices, k)
        y_pred_indices[i * k : i * k + len(top_k)] = top_k
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr
