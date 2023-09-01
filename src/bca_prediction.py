import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm, trange
from utils_sparse import *
from utils_misc import *
from utils_dense import *
from typing import Union


# Enable slow (without use of specialized numba functions) but still memory efficient implementation
SLOW = False
EPS = 1e-5


def bca_with_0approx_np(y_proba: np.ndarray, k: int, utility_func: callable, 
                                greedy_start=False, tolerance: float = 1e-5, max_iter: int = 10, 
                                shuffle_order: bool =True, seed: int = None, **kwargs):
    ni, nl = y_proba.shape

    # Initialize the prediction variable with some feasible value
    y_pred = random_at_k(y_proba, k)

    # For debug set it to first k labels
    #y_pred = np.zeros(y_proba.shape, np.float32)
    #y_pred[:,:k] = 1.0 

    # Other initializations
    # y_pred = optimal_instance_precision(y_proba, k)
    # y_pred = optimal_macro_recall(y_proba, k, marginal=np.mean(y_proba, axis=0))

    for j in range(max_iter):

        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)
        
        if greedy_start and j == 0:
            Etp = np.zeros(nl, np.float32)
            Efp = np.zeros(nl, np.float32)
            Efn = np.zeros(nl, np.float32)
        else:
            # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
            Etp = np.sum(y_pred * y_proba, axis=0)
            Efp = np.sum(y_pred * (1-y_proba), axis=0)
            Efn = np.sum((1-y_pred) * y_proba, axis=0)

        # Check expected conf matrices
        # print("Etp:", Etp.shape, type(Etp), Etp)
        # print("Efp:", Efp.shape, type(Efp), Efp)
        # print("Efn:", Efn.shape, type(Efn), Efn)

        old_utility = np.mean(utility_func(Etp / ni, Efp / ni, Efn / ni))

        for i in order:
            eta = y_proba[i, :]

            # adjust a and b locally
            if not (greedy_start and j == 0):
                Etp -= y_pred[i] * eta
                Efp -= y_pred[i] * (1-eta)
                Efn -= (1-y_pred[i]) * eta

            # calculate gain and selection
            Etpp = Etp + eta
            Efpp = Efp + (1-eta)
            Efnn = Efn + eta
            p_utility = utility_func(Etpp / ni, Efpp / ni, Efn / ni)
            n_utility = utility_func(Etp / ni, Efp / ni, Efnn / ni)
            gains = p_utility - n_utility
            top_k = np.argpartition(-gains, k)[:k]

            # update y_proba
            y_pred[i, :] = 0.0
            y_pred[i, top_k] = 1.0

            # update a and b
            Etp += y_pred[i] * eta
            Efp += y_pred[i] * (1 - eta)
            Efn += (1 - y_pred[i]) * eta

        new_utility = np.mean(utility_func(Etp / ni, Efp / ni, Efn / ni))
        print(f"  Iteration {j + 1} finished, expected score: {old_utility} -> {new_utility}")
        if new_utility <= old_utility + tolerance:
            break

    return y_pred


def bca_coverage_np(y_proba: csr_matrix, k: int, greedy_start=False, tolerance: float = 1e-5, max_iter: int = 10, 
                                   shuffle_order: bool = True, seed: int = None, filename: str = None, **kwargs):
    """
    An efficient implementation of the block coordinate-descent for coverage
    """
    ni, nl = y_proba.shape

    # initialize the prediction variable with some feasible value
    y_pred = predict_random_at_k(y_proba, k)
    y_proba = np.minimum(y_proba, 1 - 1e-5)

    for j in range(max_iter):
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)
        
        if greedy_start and j == 0:
            f = np.ones(nl, np.float32)
        else:
            f = np.product(1 - y_pred * y_proba, axis=0)
        old_cov = 1 - np.mean(f)

        for i in order:
            # adjust f locally
            f /= (1 - y_pred[i] * y_proba[i])

            # calculate gain and selection
            g = f * y_proba[i]
            top_k = np.argpartition(-g, k)[:k]

            # update y_proba
            y_pred[i, :] = 0.0
            y_pred[i, top_k] = 1.0

            # update f
            f *= (1 - y_pred[i] * y_proba[i])

        new_cov = 1 - np.mean(f)
        print(f"  Iteration {j + 1} finished, expected coverage: {old_cov} -> {new_cov}")
        if new_cov <= old_cov + tolerance:
            break
        
    return y_pred


def bca_with_0approx_csr(y_proba: csr_matrix, k: int, utility_func: callable, 
                                 greedy_start=False, tolerance: float = 1e-5, max_iter: int = 10, 
                                 shuffle_order: bool = True, seed: int = None, filename: str = None, **kwargs):

    if seed is not None:
        print(f"  Using seed: {seed}")
        np.random.seed(seed)

    # Initialize the prediction variable with some feasible value
    ni, nl = y_proba.shape
    y_pred_data, y_pred_indices, y_pred_indptr = numba_random_at_k(y_proba.indices, y_proba.indptr, ni, nl, k, seed=seed)
    
    # For debug set it to first k labels
    #y_pred_data, y_pred_indices, y_pred_indptr = numba_first_k(y_proba.data, y_proba.indices, y_proba.indptr, ni, nl, k)
    
    y_pred = construct_csr_matrix(y_pred_data, y_pred_indices, y_pred_indptr, shape=(ni, nl), sort_indices=True)


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
                Etp = calculate_tp_csr_slow(y_proba, y_pred)
                Efp = calculate_fp_csr_slow(y_proba, y_pred)
                Efn = calculate_fn_csr_slow(y_proba, y_pred)
            else:
                Etp = calculate_tp_csr(y_proba, y_pred)
                Efp = calculate_fp_csr(y_proba, y_pred)
                Efn = calculate_fn_csr(y_proba, y_pred)
        
        old_utility = np.mean(utility_func(Etp / ni, Efp / ni, Efn / ni))

        # Check expected conf matrices
        # print("Etp:", Etp.shape, type(Etp), Etp)
        # print("Efp:", Efp.shape, type(Efp), Efp)
        # print("Efn:", Efn.shape, type(Efn), Efn)

        for i in tqdm(order):
        #for i in order:
            
            if SLOW:
                eta = y_proba[i]
                dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
                
                if not (greedy_start and j == 0):
                    # Adjust local Etp, Efp, Efn
                    Etp -= y_pred[i].multiply(eta)
                    Efp -= y_pred[i].multiply(dense_ones - eta)
                    Efn -= eta.multiply(dense_ones - y_pred[i])

                Etpp = Etp + eta
                Efpp = Efp + (dense_ones - eta)
                Efnn = Efn + eta

                # Calculate gain and selection
                p_utility = utility_func(Etpp / ni, Efpp / ni, Efn / ni)
                n_utility = utility_func(Etp / ni, Efp / ni, Efnn / ni)
                gains = p_utility - n_utility
                gains = np.asarray(gains).ravel()
                top_k = np.argpartition(-gains, k)[:k]

                # Update y_proba
                y_pred.indices[y_pred.indptr[i]:y_pred.indptr[i+1]] = sorted(top_k)

                # Update Etp, Efp, Efn
                Etp += y_pred[i].multiply(eta)
                Efp += y_pred[i].multiply(dense_ones - eta)
                Efn += eta.multiply(dense_ones - y_pred[i])

            else:
                r_start, r_end = y_pred.indptr[i], y_pred.indptr[i+1]
                p_start, p_end = y_proba.indptr[i], y_proba.indptr[i+1]

                r_data = y_pred.data[r_start:r_end]
                r_indices = y_pred.indices[r_start:r_end]

                p_data = y_proba.data[p_start:p_end]
                p_indices = y_proba.indices[p_start:p_end]
                
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
                p_utility = utility_func(p_Etpp / ni, p_Efpp / ni, p_Efn / ni)
                n_utility = utility_func(p_Etp / ni, p_Efp / ni, p_Efnn / ni)
                
                # Update select labels with highest gain and update y_proba
                gains = p_utility - n_utility
                gains = np.asarray(gains).ravel()
                if gains.size > k:
                    top_k = np.argpartition(-gains, k)[:k]
                    y_pred.indices[r_start:r_end] = sorted(p_indices[top_k])
                else:
                    p_indices = np.resize(p_indices, k)
                    p_indices[gains.size:] = 0
                    y_pred.indices[r_start:r_end] = sorted(p_indices)

                # Update Etp, Efp, Efn
                data, indices = numba_sparse_vec_mul_vec(r_data, r_indices, p_data, p_indices) 
                Etp[indices] += data
                data, indices = numba_sparse_vec_mul_ones_minus_vec(r_data, r_indices, p_data, p_indices)
                Efp[indices] += data
                data, indices = numba_sparse_vec_mul_ones_minus_vec(p_data, p_indices, r_data, r_indices)
                Efn[indices] += data

        new_utility = np.mean(utility_func(Etp / ni, Efp / ni, Efn / ni))
        print(f"  Iteration {j + 1} finished, expected score: {old_utility} -> {new_utility}")
        if new_utility <= old_utility + tolerance:
            break
        
        if filename is not None:
            save_npz(f"{filename}_pred_iter_{j + 1}.npz", y_pred)
            
    return y_pred


def bca_coverage_csr(y_proba: csr_matrix, k: int, greedy_start=False, tolerance: float = 1e-5, max_iter: int = 10, 
                                  shuffle_order: bool = True, seed: int = None, filename: str = None, **kwargs):
    """
    An efficient implementation of the block coordinate-descent for coverage
    """
    if seed is not None:
        print(f"  Using seed: {seed}")
        np.random.seed(seed)

    # Initialize the prediction variable with some feasible value
    ni, nl = y_proba.shape
    y_pred_data, y_pred_indices, y_pred_indptr = numba_random_at_k(y_proba.indices, y_proba.indptr, ni, nl, k, seed=seed)
    
    # For debug set it to first k labels
    #y_pred_data, y_pred_indices, y_pred_indptr = numba_first_k(y_proba.data, y_proba.indices, y_proba.indptr, ni, nl, k)
    
    y_pred = construct_csr_matrix(y_pred_data, y_pred_indices, y_pred_indptr, shape=(ni, nl), sort_indices=True)
    y_proba.data = np.minimum(y_proba.data, 1 - 1e-5)

    for j in range(max_iter):
        
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        if greedy_start and j == 0:
            failure_prob = np.ones(nl, dtype=FLOAT_TYPE)
        else:
            failure_prob = numba_calculate_prod_1_sparse_mat_mul_ones_minus_mat(*unpack_csr_matrices(y_pred, y_proba), ni, nl)

        old_cov = 1 - np.mean(failure_prob)

        for i in tqdm(order):
        #for i in order:
            r_start, r_end = y_pred.indptr[i], y_pred.indptr[i+1]
            p_start, p_end = y_proba.indptr[i], y_proba.indptr[i+1]

            r_data = y_pred.data[r_start:r_end]
            r_indices = y_pred.indices[r_start:r_end]

            p_data = y_proba.data[p_start:p_end]
            p_indices = y_proba.indices[p_start:p_end]

            if not (greedy_start and j == 0):
                # Adjust local probablity of the failure (not covering the label)
                data, indices = numba_sparse_vec_mul_ones_minus_vec(r_data, r_indices, p_data, p_indices) 
                failure_prob[indices] /= data

            # Calculate gain and selectio
            gains = failure_prob[p_indices] * p_data
            if gains.size > k:
                top_k = np.argpartition(-gains, k)[:k]
                y_pred.indices[r_start:r_end] = sorted(p_indices[top_k])
            else:
                p_indices = np.resize(p_indices, k)
                p_indices[gains.size:] = 0
                y_pred.indices[r_start:r_end] = sorted(p_indices)

            # Update probablity of the failure (not covering the label)
            data, indices = numba_sparse_vec_mul_ones_minus_vec(r_data, r_indices, p_data, p_indices) 
            failure_prob[indices] *= data

        new_cov = 1 - np.mean(failure_prob)
        print(f"  Iteration {j + 1} finished, expected coverage: {old_cov} -> {new_cov}")
        if new_cov <= old_cov + tolerance:
            break

        if filename is not None:
            save_npz(f"{filename}_pred_iter_{j + 1}.npz", y_pred)

    return y_pred



def bca_coverage(y_proba: Union[np.ndarray, csr_matrix], k: int, tolerance: float = 1e-5, max_iter=10, seed: int = None, **kwargs):
    if isinstance(y_proba, np.ndarray):
        # Invoke original dense implementation of Erik
        return bca_coverage_np(y_proba, k, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs)
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        return bca_coverage_csr(y_proba, k, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs)
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")


def bca_with_0approx(y_proba: Union[np.ndarray, csr_matrix], k: int, utility_func: callable, 
                             tolerance: float = 1e-5, max_iter=100, seed: int = None, **kwargs):
    if isinstance(y_proba, np.ndarray):
        # Invoke original dense implementation of Erik
        return bca_with_0approx_np(y_proba, k, utility_func, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs)
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        return bca_with_0approx_csr(y_proba, k, utility_func, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs)
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")


# Implementations of functions for optimizing specific measures


def instance_precision_at_k_on_conf_matrix(tp, fp, fn, k):
    return np.asarray(tp / k).ravel()


def macro_precision_on_conf_matrix(tp, fp, fn, epsilon=1e-5):
    return np.asarray(tp / (tp + fp + epsilon)).ravel()


def macro_recall_on_conf_matrix(tp, fp, fn, epsilon=1e-5):
    return np.asarray(tp / (tp + fn + epsilon)).ravel()


def macro_fmeasure_on_conf_matrix(tp, fp, fn, beta=1.0, epsilon=1e-5):
    precision = macro_precision_on_conf_matrix(tp, fp, fn, epsilon=epsilon)
    recall = macro_recall_on_conf_matrix(tp, fp, fn, epsilon=epsilon)
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)


def block_coordinate_coverage(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return bca_coverage(y_proba, k=k, **kwargs)


def block_coordinate_macro_precision(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return bca_with_0approx(y_proba, k=k, utility_func=macro_precision_on_conf_matrix, **kwargs)


def block_coordinate_macro_recall(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return bca_with_0approx(y_proba, k=k, utility_func=macro_recall_on_conf_matrix, **kwargs)


def block_coordinate_macro_f1(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return bca_with_0approx(y_proba, k=k, utility_func=macro_fmeasure_on_conf_matrix, **kwargs)


def block_coordinate_mixed_precision(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 0.5, **kwargs):
    def mixed_precision_alpha_fn(tp, fp, fn):
        return alpha * instance_precision_at_k_on_conf_matrix(tp, fp, fn, k) + (1 - alpha) * macro_precision_on_conf_matrix(tp, fp, fn)
    return bca_with_0approx(y_proba, k=k, utility_func=mixed_precision_alpha_fn, **kwargs)


def block_coordinate_instance_prec_macro_f1(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 0.5, **kwargs):
    def mixed_precision_alpha_fn(tp, fp, fn):
        return alpha * instance_precision_at_k_on_conf_matrix(tp, fp, fn, k) + (1 - alpha) * macro_fmeasure_on_conf_matrix(tp, fp, fn)
    return bca_with_0approx(y_proba, k=k, utility_func=mixed_precision_alpha_fn, **kwargs)
