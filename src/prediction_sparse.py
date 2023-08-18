import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from numba import njit
from tqdm import tqdm, trange
from utils import *


FLOAT_TYPE=np.float32
IND_TYPE=np.int32
SLOW = False


from numba import njit
import numpy as np


@njit
def numba_first_k(ni, k):
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indices = np.zeros(ni * k, dtype=IND_TYPE)
    result_indptr = np.zeros(ni + 1, dtype=IND_TYPE)
    for i in range(ni):
        result_indices[(i * k):((i + 1) * k)] = np.arange(0, k, 1, FLOAT_TYPE)
        result_indptr[i + 1] = result_indptr[i] + k
    return result_data, result_indices, result_indptr


@njit
def numba_random_at_k(indices: np.ndarray, indptr: np.ndarray, 
                      ni: int, nl: int, k: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indices = np.zeros(ni * k, dtype=IND_TYPE)
    result_indptr = np.zeros(ni + 1, dtype=IND_TYPE)
    labels_range = np.arange(nl, dtype=IND_TYPE)
    for i in range(ni):
        row_indices = indices[indptr[i]:indptr[i+1]]
        if row_indices.size >= k:
            result_indices[i * k:(i + 1) * k] = np.random.choice(row_indices, k, replace=False)
        else:
            result_indices[i * k:(i + 1) * k] = np.random.choice(labels_range, k, replace=False)
        result_indptr[i + 1] = result_indptr[i] + k

    return result_data, result_indices, result_indptr


@njit
def numba_sparse_vec_mul_vec(a_data: np.ndarray, a_indices: np.ndarray, 
                             b_data: np.ndarray, b_indices: np.ndarray):
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
    Gives the same result as a.multiply(ones - b) where a and b are sparse vectors.
    Requires a and b to have sorted indices (in ascending order).
    """
    i = j = k = 0
    new_data_size = a_data.size + b_data.size
    new_data = np.zeros(new_data_size, dtype=FLOAT_TYPE)
    new_indices = np.zeros(new_data_size, dtype=IND_TYPE)
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
    Gives the same result as a.multiply(ones - b) where a and b are sparse matrices.
    Requires a and b to have sorted indices (in ascending order).
    """
    result = np.zeros(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        a_start, a_end = a_indptr[i], a_indptr[i+1]
        b_start, b_end = b_indptr[i], b_indptr[i+1]

        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end]
        )
        result[indices] += data

    return result


@njit
def numba_calculate_prod_1_sparse_mat_mul_ones_minus_mat(a_data, a_indices, a_indptr, b_data, b_indices, b_indptr, ni, nl):
    """
    Performs a fast multiplication of a sparse matrix a 
    with a dense matrix of ones minus other sparse matrix b and then sums the rows (axis=0).
    Gives the same result as a.multiply(ones - b) where a and b are sparse matrices.
    Requires a and b to have sorted indices (in ascending order).
    """
    result = np.ones(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        a_start, a_end = a_indptr[i], a_indptr[i+1]
        b_start, b_end = b_indptr[i], b_indptr[i+1]

        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end]
        )
        result[indices] *= data

    return result


#@njit
def argtopk(data, indices, k):
    """
    Returns the indices of the top k elements
    """
    #return indices[np.argsort(-data)[:k]]
    # argpartition is faster than sort for large datasets, but not supported by Numba due to undefined behaviours
    if data.size > k:
        top_k = indices[np.argpartition(-data, k)[:k]]
        return sorted(top_k)
    else:
        return indices


def csr_weighted_per_instance(prediction: csr_matrix, weights: np.ndarray, k: int = 5):
    # Since many numpy functions are not supported for sparse matrices
    ni, nl = prediction.shape
    data, indices, indptr = numba_weighted_per_instance(prediction.data, prediction.indices, prediction.indptr, weights, ni, nl, k)
    return csr_matrix((data, indices, indptr), shape=prediction.shape)


#@njit
def numba_weighted_per_instance(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, 
                                weights: np.ndarray, ni: int, nl: int, k: int):
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indices = np.zeros(ni * k, dtype=IND_TYPE)
    result_indptr = np.zeros(ni + 1, dtype=IND_TYPE)

    # This can be done in parallel, but Numba parallelism seems to not work well here
    for i in range(ni):
        row_data = data[indptr[i]:indptr[i+1]]
        row_indices = indices[indptr[i]:indptr[i+1]]
        row_weights = weights[row_indices].reshape(-1) * row_data
        top_k = argtopk(row_weights, row_indices, k)
        result_indices[i * k:i * k + len(top_k)] = top_k
        #result_indices[i * k:(i + 1) * k] = top_k
        result_indptr[i + 1] = result_indptr[i] + k

    return result_data, result_indices, result_indptr


def calculate_etp_slow(result: csr_matrix, probabilities: csr_matrix):
    return (result.multiply(probabilities)).sum(axis=0)


def calculate_etp(result: csr_matrix, probabilities: csr_matrix):
    ni, nl = probabilities.shape
    Etp = np.zeros(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        r_start, r_end = result.indptr[i], result.indptr[i+1]
        p_start, p_end = probabilities.indptr[i], probabilities.indptr[i+1]

        data, indices = numba_sparse_vec_mul_vec(
            result.data[r_start:r_end],
            result.indices[r_start:r_end],
            probabilities.data[p_start:p_end],
            probabilities.indices[p_start:p_end]
        )
        Etp[indices] += data

    return Etp


# This is a bit slow, TODO: make it faster (drop multiply and use custom method)
def calculate_efp_slow(result: csr_matrix, probabilities: csr_matrix):
    ni, nl = probabilities.shape
    Efp = np.zeros(nl, dtype=FLOAT_TYPE)
    dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        Efp += result[i].multiply(dense_ones - probabilities[i])
    return Efp


def calculate_efp(result: csr_matrix, probabilities: csr_matrix):
    ni, nl = probabilities.shape
    return numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(*unpack_csr_matrices(result, probabilities), ni, nl)


# This is a bit slow, TODO: make it faster (drop multiply and use custom method)
def calculate_efn_slow(result: csr_matrix, probabilities: csr_matrix):
    ni, nl = probabilities.shape
    Efn = np.zeros(nl, dtype=FLOAT_TYPE)
    dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        Efn += probabilities[i].multiply(dense_ones - result[i]) 
    
    return Efn


def calculate_efn(result: csr_matrix, probabilities: csr_matrix):
    ni, nl = probabilities.shape
    return numba_calculate_sum_0_sparse_mat_mul_ones_minus_mat(*unpack_csr_matrices(probabilities, result), ni, nl)


def csr_macro_population_cm_risk(probabilities: csr_matrix, k: int, measure_func: callable, 
                                 greedy_start=False, tolerance: float = 1e-5, max_iter: int = 10, 
                                 shuffle_order: bool = True, seed: int = None, filename: str = None, **kwargs):

    if seed is not None:
        print(f"  Using seed: {seed}")
        np.random.seed(seed)

    # Initialize the prediction variable with some feasible value
    ni, nl = probabilities.shape
    result_data, result_indices, result_indptr = numba_random_at_k(probabilities.indices, probabilities.indptr, ni, nl, k, seed=seed)
    #result_data, result_indices, result_indptr = numba_weighted_per_instance(probabilities.data, probabilities.indices, probabilities.indptr, np.ones((nl,), dtype=np.float32), ni, nl, k)
    
    # For debug set it to first k labels
    #result_data, result_indices, result_indptr = numba_first_k(probabilities.data, probabilities.indices, probabilities.indptr, ni, nl, k)
    
    result = construct_csr_matrix(result_data, result_indices, result_indptr, shape=(ni, nl), sort_indices=True)
    iters = 0

    for j in range(max_iter):

        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
        # In this variant they will be all np.matrix with shape (1, nl)
        if greedy_start and j == 0:
            Etp = np.zeros(nl, FLOAT_TYPE)
            Efp = np.zeros(nl, FLOAT_TYPE)
            Efn = np.zeros(nl, FLOAT_TYPE)
        else:
            if SLOW:
                Etp = calculate_etp_slow(result, probabilities)
                Efp = calculate_efp_slow(result, probabilities)
                Efn = calculate_efn_slow(result, probabilities)
            else:
                Etp = calculate_etp(result, probabilities)
                Efp = calculate_efp(result, probabilities)
                Efn = calculate_efn(result, probabilities)
        
        old_score = np.mean(measure_func(Etp / ni, Efp / ni, Efn / ni))

        # Check expected conf matrices
        # print("Etp:", Etp.shape, type(Etp), Etp)
        # print("Efp:", Efp.shape, type(Efp), Efp)
        # print("Efn:", Efn.shape, type(Efn), Efn)

        for i in tqdm(order):
        #for i in order:
            
            if SLOW:
                eta = probabilities[i]
                dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
                
                if not (greedy_start and j == 0):
                    # Adjust local Etp, Efp, Efn
                    Etp -= result[i].multiply(eta)
                    Efp -= result[i].multiply(dense_ones - eta)
                    Efn -= eta.multiply(dense_ones - result[i])

                Etpp = Etp + eta
                Efpp = Efp + (dense_ones - eta)
                Efnn = Efn + eta

                # Calculate gain and selection
                p_score = measure_func(Etpp / ni, Efpp / ni, Efn / ni)
                n_score = measure_func(Etp / ni, Efp / ni, Efnn / ni)
                gains = p_score - n_score
                gains = np.asarray(gains).ravel()
                top_k = np.argpartition(-gains, k)[:k]

                # Update predictions
                result.indices[result.indptr[i]:result.indptr[i+1]] = sorted(top_k)

                # Update Etp, Efp, Efn
                Etp += result[i].multiply(eta)
                Efp += result[i].multiply(dense_ones - eta)
                Efn += eta.multiply(dense_ones - result[i])

            else:
                r_start, r_end = result.indptr[i], result.indptr[i+1]
                p_start, p_end = probabilities.indptr[i], probabilities.indptr[i+1]

                r_data = result.data[r_start:r_end]
                r_indices = result.indices[r_start:r_end]

                p_data = probabilities.data[p_start:p_end]
                p_indices = probabilities.indices[p_start:p_end]
                
                if not (greedy_start and j == 0):
                    # Adjust local Etp, Efp, Efn
                    data, indices = numba_sparse_vec_mul_vec(r_data, r_indices, p_data, p_indices) 
                    Etp[indices] -= data
                    data, indices = numba_sparse_vec_mul_ones_minus_vec(r_data, r_indices, p_data, p_indices)
                    Efp[indices] -= data
                    data, indices = numba_sparse_vec_mul_ones_minus_vec(p_data, p_indices, r_data, r_indices)
                    Efn[indices] -= data

                p_Etp = Etp[p_indices]
                p_Efp = Efp[p_indices]
                p_Efn = Efn[p_indices]

                p_Etpp = Etp[p_indices] + p_data
                p_Efpp = Efp[p_indices] + (1 - p_data)
                p_Efnn = Efn[p_indices] + p_data

                # Calculate gain and selection
                p_score = measure_func(p_Etpp / ni, p_Efpp / ni, p_Efn / ni)
                n_score = measure_func(p_Etp / ni, p_Efp / ni, p_Efnn / ni)
                
                # Update select labels with highest gain and update predictions
                gains = p_score - n_score
                gains = np.asarray(gains).ravel()
                if gains.size > k:
                    top_k = np.argpartition(-gains, k)[:k]
                    result.indices[r_start:r_end] = sorted(p_indices[top_k])
                else:
                    p_indices = np.resize(p_indices, k)
                    p_indices[gains.size:] = 0
                    result.indices[r_start:r_end] = sorted(p_indices)

                # Update Etp, Efp, Efn
                data, indices = numba_sparse_vec_mul_vec(r_data, r_indices, p_data, p_indices) 
                Etp[indices] += data
                data, indices = numba_sparse_vec_mul_ones_minus_vec(r_data, r_indices, p_data, p_indices)
                Efp[indices] += data
                data, indices = numba_sparse_vec_mul_ones_minus_vec(p_data, p_indices, r_data, r_indices)
                Efn[indices] += data
        
        iters = j
        new_score = np.mean(measure_func(Etp / ni, Efp / ni, Efn / ni))
        print(f"  Iteration {j + 1} finished, expected score: {old_score} -> {new_score}")
        if new_score <= old_score + tolerance:
            break
        
        if filename is not None:
            save_npz(f"{filename}_pred_iter_{j + 1}.npz", result)
            
    return result, iters


def csr_block_coordinate_coverage(probabilities: csr_matrix, k: int, greedy_start=False, tolerance: float = 1e-5, max_iter: int = 10, 
                                  shuffle_order: bool = True, seed: int = None, filename: str = None, **kwargs):
    """
    An efficient implementation of the block coordinate-descent for coverage
    """
    if seed is not None:
        print(f"  Using seed: {seed}")
        np.random.seed(seed)

    # Initialize the prediction variable with some feasible value
    ni, nl = probabilities.shape
    result_data, result_indices, result_indptr = numba_random_at_k(probabilities.indices, probabilities.indptr, ni, nl, k, seed=seed)
    
    # For debug set it to first k labels
    #result_data, result_indices, result_indptr = numba_first_k(probabilities.data, probabilities.indices, probabilities.indptr, ni, nl, k)
    
    result = construct_csr_matrix(result_data, result_indices, result_indptr, shape=(ni, nl), sort_indices=True)
    probabilities.data = np.minimum(probabilities.data, 1 - 1e-5)
    iters = 0

    for j in range(max_iter):
        
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        if greedy_start and j == 0:
            failure_prob = np.ones(nl, dtype=FLOAT_TYPE)
        else:
            failure_prob = numba_calculate_prod_1_sparse_mat_mul_ones_minus_mat(*unpack_csr_matrices(result, probabilities), ni, nl)

        old_cov = 1 - np.mean(failure_prob)

        for i in tqdm(order):
        #for i in order:
            r_start, r_end = result.indptr[i], result.indptr[i+1]
            p_start, p_end = probabilities.indptr[i], probabilities.indptr[i+1]

            r_data = result.data[r_start:r_end]
            r_indices = result.indices[r_start:r_end]

            p_data = probabilities.data[p_start:p_end]
            p_indices = probabilities.indices[p_start:p_end]

            if not (greedy_start and j == 0):
                # Adjust local probablity of the failure (not covering the label)
                data, indices = numba_sparse_vec_mul_ones_minus_vec(r_data, r_indices, p_data, p_indices) 
                failure_prob[indices] /= data

            # Calculate gain and selectio
            gains = failure_prob[p_indices] * p_data
            if gains.size > k:
                top_k = np.argpartition(-gains, k)[:k]
                result.indices[r_start:r_end] = sorted(p_indices[top_k])
            else:
                p_indices = np.resize(p_indices, k)
                p_indices[gains.size:] = 0
                result.indices[r_start:r_end] = sorted(p_indices)

            # Update probablity of the failure (not covering the label)
            data, indices = numba_sparse_vec_mul_ones_minus_vec(r_data, r_indices, p_data, p_indices) 
            failure_prob[indices] *= data

        iters = j
        new_cov = 1 - np.mean(failure_prob)
        print(f"  Iteration {j + 1} finished, expected coverage: {old_cov} -> {new_cov}")
        if new_cov <= old_cov + tolerance:
            break

        if filename is not None:
            save_npz(f"{filename}_pred_iter_{j + 1}.npz", result)

    return result, iters
