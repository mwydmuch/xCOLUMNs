import numpy as np
from scipy.sparse import csr_matrix
from numba import njit


FLOAT_TYPE=np.float32
INT_TYPE=np.int32


def unpack_csr_matrix(matrix: csr_matrix):
    return matrix.data, matrix.indices, matrix.indptr


def unpack_csr_matrices(*matrices):
    y_pred = []
    for m in matrices:
        y_pred.extend(unpack_csr_matrix(m))
    return y_pred


def construct_csr_matrix(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, dtype=None, shape=None, sort_indices=False):
    mat = csr_matrix((data, indices, indptr), dtype=dtype, shape=shape)
    if sort_indices:
        mat.sort_indices()
    return mat


@njit
def numba_first_k(ni, k):
    y_pred_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(ni * k, dtype=INT_TYPE)
    y_pred_indptr = np.zeros(ni + 1, dtype=INT_TYPE)
    for i in range(ni):
        y_pred_indices[(i * k):((i + 1) * k)] = np.arange(0, k, 1, FLOAT_TYPE)
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k
    return y_pred_data, y_pred_indices, y_pred_indptr


@njit
def numba_random_at_k(indices: np.ndarray, indptr: np.ndarray, 
                      ni: int, nl: int, k: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    y_pred_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(ni * k, dtype=INT_TYPE)
    y_pred_indptr = np.zeros(ni + 1, dtype=INT_TYPE)
    labels_range = np.arange(nl, dtype=INT_TYPE)
    for i in range(ni):
        row_indices = indices[indptr[i]:indptr[i+1]]
        if row_indices.size >= k:
            y_pred_indices[i * k:(i + 1) * k] = np.random.choice(row_indices, k, replace=False)
        else:
            y_pred_indices[i * k:(i + 1) * k] = np.random.choice(labels_range, k, replace=False)
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


@njit
def numba_sparse_vec_mul_vec(a_data: np.ndarray, a_indices: np.ndarray, 
                             b_data: np.ndarray, b_indices: np.ndarray):
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
        #print(i, j, k, a_indices[i], b_indices[j], a_indices.size, b_indices.size)
        if a_indices[i] < b_indices[j]:
            i+=1
        elif a_indices[i] == b_indices[j]: 
            new_data[k] = a_data[i] * b_data[j]
            new_indices[k] = a_indices[i]
            k+=1
            i+=1
            j+=1
        else: 
            j+=1
    return new_data[:k], new_indices[:k]


@njit
def numba_sparse_vec_mul_ones_minus_vec(a_data: np.ndarray, a_indices: np.ndarray, 
                                        b_data: np.ndarray, b_indices: np.ndarray):
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
        #print(i, j, k, a_indices[i], b_indices[j], a_indices.size, b_indices.size)
        if j >= b_indices.size or a_indices[i] < b_indices[j]:
            new_data[k] = a_data[i]
            new_indices[k] = a_indices[i]
            k+=1
            i+=1
        elif a_indices[i] == b_indices[j]: 
            new_data[k] = a_data[i] * (1 - b_data[j])
            new_indices[k] = a_indices[i]
            k+=1
            i+=1
            j+=1
        else: 
            j+=1
    return new_data[:k], new_indices[:k]


@njit
def numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(a_data, a_indices, a_indptr, b_data, b_indices, b_indptr, ni, nl):
    """
    Performs a fast multiplication of a sparse matrix a 
    with a dense matrix of ones minus other sparse matrix b and then sums the rows (axis=0).
    Gives the same y_pred as a.multiply(ones - b) where a and b are sparse matrices.
    Requires a and b to have sorted indices (in ascending order).
    """
    y_pred = np.zeros(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        a_start, a_end = a_indptr[i], a_indptr[i+1]
        b_start, b_end = b_indptr[i], b_indptr[i+1]

        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end]
        )
        y_pred[indices] += data

    return y_pred


@njit
def numba_calculate_prod_1_sparse_mat_mul_ones_minus_mat(a_data, a_indices, a_indptr, b_data, b_indices, b_indptr, ni, nl):
    """
    Performs a fast multiplication of a sparse matrix a 
    with a dense matrix of ones minus other sparse matrix b and then sums the rows (axis=0).
    Gives the same y_pred as a.multiply(ones - b) where a and b are sparse matrices.
    Requires a and b to have sorted indices (in ascending order).
    """
    y_pred = np.ones(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        a_start, a_end = a_indptr[i], a_indptr[i+1]
        b_start, b_end = b_indptr[i], b_indptr[i+1]

        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end]
        )
        y_pred[indices] *= data

    return y_pred


#@njit
def numba_argtopk(data, indices, k):
    """
    Returns the indices of the top k elements
    """
    #return indices[np.argsort(-data)[:k]]
    # argpartition is faster than sort for large datasets, but not supported by Numba due to undefined behaviours
    # To enable njit, we shoud implement our own argpartition (TODO)
    if data.size > k:
        top_k = indices[np.argpartition(-data, k)[:k]]
        return sorted(top_k)
    else:
        return indices


#@njit
def numba_weighted_per_instance(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, 
                                weights: np.ndarray, ni: int, nl: int, k: int):
    y_pred_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    y_pred_indices = np.zeros(ni * k, dtype=INT_TYPE)
    y_pred_indptr = np.zeros(ni + 1, dtype=INT_TYPE)

    # This can be done in parallel, but Numba parallelism seems to not work well here
    for i in range(ni):
        row_data = data[indptr[i]:indptr[i+1]]
        row_indices = indices[indptr[i]:indptr[i+1]]
        row_weights = weights[row_indices].reshape(-1) * row_data
        top_k = numba_argtopk(row_weights, row_indices, k)
        y_pred_indices[i * k:i * k + len(top_k)] = top_k
        #y_pred_indices[i * k:(i + 1) * k] = top_k
        y_pred_indptr[i + 1] = y_pred_indptr[i] + k

    return y_pred_data, y_pred_indices, y_pred_indptr


def calculate_tp_csr_slow(y_proba: csr_matrix, y_pred: csr_matrix):
    return (y_pred.multiply(y_proba)).sum(axis=0)


def calculate_tp_csr(y_proba: csr_matrix, y_pred: csr_matrix):
    """
    Calculate 0 approx. of true positives or true number of true positives if y_proba = y_true
    """
    ni, nl = y_proba.shape
    Etp = np.zeros(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        r_start, r_end = y_pred.indptr[i], y_pred.indptr[i+1]
        p_start, p_end = y_proba.indptr[i], y_proba.indptr[i+1]

        data, indices = numba_sparse_vec_mul_vec(
            y_pred.data[r_start:r_end],
            y_pred.indices[r_start:r_end],
            y_proba.data[p_start:p_end],
            y_proba.indices[p_start:p_end]
        )
        Etp[indices] += data

    return Etp


# This is a bit slow, TODO: make it faster (drop multiply and use custom method)
def calculate_fp_csr_slow(y_proba: csr_matrix, y_pred: csr_matrix):
    ni, nl = y_proba.shape
    Efp = np.zeros(nl, dtype=FLOAT_TYPE)
    dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        Efp += y_pred[i].multiply(dense_ones - y_proba[i])
    return Efp


def calculate_fp_csr(y_proba: csr_matrix, y_pred: csr_matrix):
    """
    Calculate 0 approx. of false positives or true number of false positives if y_proba = y_true
    """
    ni, nl = y_proba.shape
    return numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(*unpack_csr_matrices(y_pred, y_proba), ni, nl)


# This is a bit slow, TODO: make it faster (drop multiply and use custom method)
def calculate_fn_csr_slow(y_proba: csr_matrix, y_pred: csr_matrix):
    ni, nl = y_proba.shape
    Efn = np.zeros(nl, dtype=FLOAT_TYPE)
    dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        Efn += y_proba[i].multiply(dense_ones - y_pred[i]) 
    
    return Efn


def calculate_fn_csr(y_proba: csr_matrix, y_pred: csr_matrix):
    """
    Calculate 0 approx. of false negatives or true number of false negatives if y_proba = y_true
    """
    ni, nl = y_proba.shape
    return numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(*unpack_csr_matrices(y_proba, y_pred), ni, nl)
