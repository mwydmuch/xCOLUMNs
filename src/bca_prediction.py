import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm, trange
from weighted_prediction import predict_top_k
from utils_sparse import *
from utils_misc import *
from utils_dense import *
from typing import Union
from default_types import FLOAT_TYPE, INT_TYPE


# Enable slow (without use of specialized numba functions) but still memory efficient implementation (for debugging)
SLOW = False
EPS = 1e-6


def get_utility_aggregation_func(utility_aggregation: str):
    if utility_aggregation == "mean":
        return np.mean
    elif utility_aggregation == "sum":
        return np.sum
    else:
        raise ValueError(f"Unsupported utility aggregation function: {utility_aggregation}, must be either 'mean' or 'sum'")
    

def calculate_objective(Etp, Efp, Efn, Etn, utility_func, utility_aggregation_func):
    if callable(utility_func):
        return utility_aggregation_func(utility_func(Etp, Efp, Efn, Etn))
    else:
        return utility_aggregation_func([f(Etp, Efp, Efn, Etn) for f in utility_func])
    

def bca_with_0approx_np_step(y_proba_i, y_pred_i, Etp, Efp, Efn, Etn, ni, k, utility_func, greedy = False):
    # Adjust local confusion matrix
    if not greedy:
        Etp -= y_pred_i * y_proba_i
        Efp -= y_pred_i * (1 - y_proba_i)
        Efn -= (1 - y_pred_i) * y_proba_i
        Etn -= (1 - y_pred_i) * (1 - y_proba_i)

    # Calculate gain and selection
    Etpp = Etp + y_proba_i
    Efpp = Efp + (1 - y_proba_i)
    Efnn = Efn + y_proba_i
    Etnn = Etn + (1 - y_proba_i)
    if callable(utility_func):
        p_utility = utility_func(Etpp / ni, Efpp / ni, Efn / ni, Etn / ni)
        n_utility = utility_func(Etp / ni, Efp / ni, Efnn / ni, Etnn / ni)
    else:
        p_utility = np.array([f(Etpp / ni, Efpp / ni, Efn / ni, Etn / ni) for f in utility_func])
        n_utility = np.array([f(Etp / ni, Efp / ni, Efnn / ni, Etnn / ni) for f in utility_func])
    gains = p_utility - n_utility
    top_k = np.argpartition(-gains, k)[:k]

    # Update prediction
    y_pred_i[:] = 0.0
    y_pred_i[top_k] = 1.0

    # Update local confusion matrix
    Etp += y_pred_i * y_proba_i
    Efp += y_pred_i * (1 - y_proba_i)
    Efn += (1 - y_pred_i) * y_proba_i
    Etn += (1 - y_pred_i) * (1 - y_proba_i)


def bca_with_0approx_np(
    y_proba: np.ndarray,
    k: int,
    utility_func: Union[callable, list[callable]],
    utility_aggregation: str = "mean",  # "mean" or "sum"
    greedy_start=False,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray] ="random",  # "random", "topk", or np.ndarray
    max_iter: int = 100,
    shuffle_order: bool = True,
    seed: int = None,
    filename: str = None,
    **kwargs,
):
    if seed is not None:
        np.random.seed(seed)

    utility_aggregation_func = get_utility_aggregation_func(utility_aggregation)

    ni, nl = y_proba.shape

    # Initialize the prediction variable with some feasible value
    if init_y_pred == "random":
        y_pred = predict_random_at_k(y_proba, k)
    elif init_y_pred == "topk":
        y_pred = predict_top_k(y_proba, k)
    else:
        y_pred = init_y_pred
        assert y_pred.shape == (ni, nl)

    meta = {"utilities": []}

    for j in range(1, max_iter + 1):
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        if greedy_start and j == 0:
            Etp = np.zeros(nl, dtype=FLOAT_TYPE)
            Efp = np.zeros(nl, dtype=FLOAT_TYPE)
            Efn = np.zeros(nl, dtype=FLOAT_TYPE)
            Etn = np.zeros(nl, dtype=FLOAT_TYPE)
        else:
            # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
            Etp = np.sum(y_pred * y_proba, axis=0)
            Efp = np.sum(y_pred * (1 - y_proba), axis=0)
            Efn = np.sum((1 - y_pred) * y_proba, axis=0)
            Etn = np.full(nl, ni, dtype=FLOAT_TYPE) - Etp - Efp - Efn  # All sum to ni

        old_utility = calculate_objective(Etp / ni, Efp / ni, Efn / ni, Etn / ni, utility_func, utility_aggregation_func)

        for i in order:
            bca_with_0approx_np_step(y_proba[i, :], y_pred[i, :], Etp, Efp, Efn, Etn, ni, k, utility_func, greedy=(greedy_start and j == 0))
            
        new_utility = calculate_objective(Etp / ni, Efp / ni, Efn / ni, Etn / ni, utility_func, utility_aggregation_func)

        meta["utilities"].append(new_utility)
        print(
            f"  Iteration {j} finished, expected score: {old_utility} -> {new_utility}"
        )
        if new_utility <= old_utility + tolerance:
            break

        if filename is not None:
            save_npz(f"{filename}_pred_iter_{j}.npz", y_pred)

    meta["iters"] = j
    return y_pred, meta



def bca_with_0approx_csr_step(y_proba, y_pred, i, Etp, Efp, Efn, Etn, ni, k, utility_func, greedy=False):
    r_start, r_end = y_pred.indptr[i], y_pred.indptr[i + 1]
    p_start, p_end = y_proba.indptr[i], y_proba.indptr[i + 1]

    r_data = y_pred.data[r_start:r_end]
    r_indices = y_pred.indices[r_start:r_end]

    p_data = y_proba.data[p_start:p_end]
    p_indices = y_proba.indices[p_start:p_end]

    # Adjust local confusion matrix
    if not greedy:
        Etn -= 1 
        data, indices = numba_sparse_vec_mul_vec(
            r_data, r_indices, p_data, p_indices
        )
        Etp[indices] -= data
        Etn[indices] += data
        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            r_data, r_indices, p_data, p_indices
        )
        Efp[indices] -= data
        Efn[indices] += data
        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            p_data, p_indices, r_data, r_indices
        )
        Efn[indices] -= data
        Efp[indices] += data

    n_Etp = Etp[p_indices]
    n_Efp = Efp[p_indices]
    p_Efn = Efn[p_indices]
    p_Etn = Etn[p_indices]

    p_Etpp = Etp[p_indices] + p_data
    p_Efpp = Efp[p_indices] + (1 - p_data)
    n_Efnn = Efn[p_indices] + p_data
    n_Etnn = Etn[p_indices] + (1 - p_data)

    # Calculate gain and selection
    if callable(utility_func):
        p_utility = utility_func(p_Etpp / ni, p_Efpp / ni, p_Efn / ni, p_Etn / ni)
        n_utility = utility_func(n_Etp / ni, n_Efp / ni, n_Efnn / ni, n_Etnn / ni)
    else:
        p_utility = np.array([f(p_Etpp / ni, p_Efpp / ni, p_Efn / ni, p_Etn / ni) for f in utility_func])
        n_utility = np.array([f(n_Etp / ni, n_Efp / ni, n_Efnn / ni, n_Etnn / ni) for f in utility_func])

    # Update select labels with highest gain and update prediction
    gains = p_utility - n_utility
    gains = np.asarray(gains).ravel()
    if gains.size > k:
        top_k = np.argpartition(-gains, k)[:k]
        y_pred.indices[r_start:r_end] = sorted(p_indices[top_k])
    else:
        p_indices = np.resize(p_indices, k)
        p_indices[gains.size :] = 0
        y_pred.indices[r_start:r_end] = sorted(p_indices)
    
    # Update local confusion matrix
    Etn += 1 
    data, indices = numba_sparse_vec_mul_vec(
        r_data, r_indices, p_data, p_indices
    )
    Etp[indices] += data
    Etn[indices] -= data 
    data, indices = numba_sparse_vec_mul_ones_minus_vec(
        r_data, r_indices, p_data, p_indices
    )
    Efp[indices] += data
    Etn[indices] -= data
    data, indices = numba_sparse_vec_mul_ones_minus_vec(
        p_data, p_indices, r_data, r_indices
    )
    Efn[indices] += data
    Etn[indices] -= data
    

def bca_with_0approx_csr(
    y_proba: csr_matrix,
    k: int,
    utility_func: Union[callable, list[callable]],
    utility_aggregation: str = "mean",  # "mean" or "sum"
    greedy_start=False,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray] = "random",  # "random", "topk", or np.ndarray
    max_iter: int = 100,
    shuffle_order: bool = True,
    seed: int = None,
    verbose: bool = False,
    filename: str = None,
    **kwargs,
):
    if seed is not None:
        np.random.seed(seed)

    utility_aggregation_func = get_utility_aggregation_func(utility_aggregation)

    ni, nl = y_proba.shape

    # Initialize the prediction variable with some feasible value
    if init_y_pred == "random":
        y_pred_data, y_pred_indices, y_pred_indptr = numba_random_at_k(
            y_proba.indices, y_proba.indptr, ni, nl, k, seed=seed
        )
        y_pred = construct_csr_matrix(
            y_pred_data, y_pred_indices, y_pred_indptr, shape=(ni, nl), sort_indices=True
        )
    elif init_y_pred == "topk":
        y_pred = predict_top_k(y_proba, k, include_meta=False)
    else:
        y_pred = init_y_pred
    
    assert y_pred.shape == (ni, nl)

    meta = {"utilities": []}

    for j in range(1, max_iter + 1):
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
        # In this variant they will be all np.matrix with shape (1, nl)
        if greedy_start and j == 0:
            Etp = np.zeros(nl, dtype=FLOAT_TYPE)
            Efp = np.zeros(nl, dtype=FLOAT_TYPE)
            Efn = np.zeros(nl, dtype=FLOAT_TYPE)
            Etn = np.zeros(nl, dtype=FLOAT_TYPE)
        else:
            Etp = calculate_tp_csr(y_proba, y_pred)
            Efp = calculate_fp_csr(y_proba, y_pred)
            Efn = calculate_fn_csr(y_proba, y_pred)
            Etn = np.full(nl, ni, dtype=FLOAT_TYPE) - Etp - Efp - Efn  # All sum to ni
        
            assert Etp.shape == (nl,)
            assert Efp.shape == (nl,)
            assert Efn.shape == (nl,)
            assert Etn.shape == (nl,)
    
        old_utility = calculate_objective(Etp / ni, Efp / ni, Efn / ni, Etn / ni, utility_func, utility_aggregation_func)

        for i in tqdm(order):
            # print(y_pred[i], y_pred[i].shape)
            # print("---")
            bca_with_0approx_csr_step(y_proba, y_pred, i, Etp, Efp, Efn, Etn, ni, k, utility_func, greedy=(greedy_start and j == 0))
            # print(y_pred[i], y_pred[i].shape)
            # print("---")
            # input()

        assert Etp.shape == (nl,)
        assert Efp.shape == (nl,)
        assert Efn.shape == (nl,)
        assert Etn.shape == (nl,)

        Etp = calculate_tp_csr(y_proba, y_pred)
        Efp = calculate_fp_csr(y_proba, y_pred)
        Efn = calculate_fn_csr(y_proba, y_pred)
        Etn = np.full(nl, ni, dtype=FLOAT_TYPE) - Etp - Efp - Efn  # All sum to ni

        new_utility = calculate_objective(Etp / ni, Efp / ni, Efn / ni, Etn / ni, utility_func, utility_aggregation_func)
        meta["utilities"].append(new_utility)
        print(
            f"  Iteration {j} finished, expected score: {old_utility} -> {new_utility}"
        )
        if new_utility <= old_utility + tolerance:
            break

        if filename is not None:
            save_npz(f"{filename}_pred_iter_{j}.npz", y_pred)

    meta["iters"] = j
    return y_pred, meta


def bca_with_0approx(
    y_proba: csr_matrix,
    k: int,
    utility_func: Union[callable, list[callable]],
    greedy_start=False,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray] = "random",  # "random", "topk", or np.ndarray
    max_iter: int = 100,
    shuffle_order: bool = True,
    seed: int = None,
    filename: str = None,
    **kwargs,
):
    """
    TODO: Add docstring

    BCA with 0-approximation here uses tp, fp, fn, tn matrics parametrization of the confussion matrix,
    as opposed to algorithms presented in the final version of the paper, which use t, q, p parametrization.
    However both algorithms are equivalent.
    """
    if isinstance(y_proba, np.ndarray):
        # Invoke implementation for dense matrices
        return bca_with_0approx_np(
            y_proba,
            k,
            utility_func,
            greedy_start=greedy_start,
            tolerance=tolerance,
            init_y_pred=init_y_pred,
            max_iter=max_iter,
            shuffle_order=shuffle_order,
            seed=seed,
            filename=filename,
            **kwargs,
        )
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        return bca_with_0approx_csr(
            y_proba,
            k,
            utility_func,
            greedy_start=greedy_start,
            tolerance=tolerance,
            init_y_pred=init_y_pred,
            max_iter=max_iter,
            shuffle_order=shuffle_order,
            seed=seed,
            filename=filename,
            **kwargs,
        )
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")


# Implementations of specialized BCA for coverage measure

def bca_coverage_np(
    y_proba: csr_matrix,
    k: int,
    alpha: float = 1,
    greedy_start=False,
    tolerance: float = 1e-6,
    max_iter: int = 100,
    shuffle_order: bool = True,
    seed: int = None,
    filename: str = None,
    **kwargs,
):
    ni, nl = y_proba.shape

    # initialize the prediction variable with some feasible value
    y_pred = predict_random_at_k(y_proba, k)
    y_proba = np.minimum(y_proba, 1 - EPS)

    meta = {"utilities": []}

    for j in range(1, max_iter + 1):
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        if greedy_start and j == 0:
            f = np.ones(nl, np.float32)
        else:
            f = np.product(1 - y_pred * y_proba, axis=0)
        
        old_cov = 1 - np.mean(f)
        if alpha < 1:
            old_cov = alpha * old_cov + (1 - alpha) * (np.sum(y_pred * y_proba, axis=0) / ni / k).mean()

        for i in order:
            # adjust f locally
            f /= 1 - y_pred[i] * y_proba[i]

            # calculate gain and selection
            g = f * y_proba[i]
            if alpha < 1:
                g = alpha * g + (1 - alpha) * y_proba[i] / k
            top_k = np.argpartition(-g, k)[:k]

            # update y_proba
            y_pred[i, :] = 0.0
            y_pred[i, top_k] = 1.0

            # update f
            f *= 1 - y_pred[i] * y_proba[i]

        new_cov = 1 - np.mean(f)
        if alpha < 1:
            new_cov = alpha * new_cov + (1 - alpha) * (np.sum(y_pred * y_proba, axis=0) / ni / k).mean()

        meta["utilities"].append(new_cov)
        print(
            f"  Iteration {j} finished, expected coverage: {old_cov} -> {new_cov}"
        )
        if new_cov <= old_cov + tolerance:
            break

    meta["iters"] = j
    return y_pred, meta


def bca_coverage_csr(
    y_proba: csr_matrix,
    k: int,
    alpha: float = 1,
    greedy_start=False,
    tolerance: float = 1e-6,
    max_iter: int = 100,
    shuffle_order: bool = True,
    seed: int = None,
    filename: str = None,
    **kwargs,
):
    """
    An efficient implementation of the block coordinate-descent for coverage
    """
    if seed is not None:
        print(f"  Using seed: {seed}")
        np.random.seed(seed)

    # Initialize the prediction variable with some feasible value
    ni, nl = y_proba.shape
    y_pred_data, y_pred_indices, y_pred_indptr = numba_random_at_k(
        y_proba.indices, y_proba.indptr, ni, nl, k, seed=seed
    )

    # For debug set it to first k labels
    # y_pred_data, y_pred_indices, y_pred_indptr = numba_first_k(y_proba.data, y_proba.indices, y_proba.indptr, ni, nl, k)

    y_pred = construct_csr_matrix(
        y_pred_data, y_pred_indices, y_pred_indptr, shape=(ni, nl), sort_indices=True
    )
    y_proba.data = np.minimum(y_proba.data, 1 - EPS)

    meta = {"utilities": []}

    for j in range(1, max_iter + 1):
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        if greedy_start and j == 0:
            failure_prob = np.ones(nl, dtype=FLOAT_TYPE)
        else:
            failure_prob = numba_calculate_prod_1_sparse_mat_mul_ones_minus_mat(
                *unpack_csr_matrices(y_pred, y_proba), ni, nl
            )

        old_cov = 1 - np.mean(failure_prob)
        if alpha < 1:
            old_cov = alpha * old_cov + (1 - alpha) * np.asarray(calculate_tp_csr(y_proba, y_pred) / ni / k).ravel().mean()

        for i in tqdm(order):
            # for i in order:
            r_start, r_end = y_pred.indptr[i], y_pred.indptr[i + 1]
            p_start, p_end = y_proba.indptr[i], y_proba.indptr[i + 1]

            r_data = y_pred.data[r_start:r_end]
            r_indices = y_pred.indices[r_start:r_end]

            p_data = y_proba.data[p_start:p_end]
            p_indices = y_proba.indices[p_start:p_end]

            if not (greedy_start and j == 0):
                # Adjust local probablity of the failure (not covering the label)
                data, indices = numba_sparse_vec_mul_ones_minus_vec(
                    r_data, r_indices, p_data, p_indices
                )
                failure_prob[indices] /= data

            # Calculate gain and selectio
            gains = failure_prob[p_indices] * p_data
            if alpha < 1:
                gains = alpha * gains + (1 - alpha) * p_data / k
            if gains.size > k:
                top_k = np.argpartition(-gains, k)[:k]
                y_pred.indices[r_start:r_end] = sorted(p_indices[top_k])
            else:
                p_indices = np.resize(p_indices, k)
                p_indices[gains.size :] = 0
                y_pred.indices[r_start:r_end] = sorted(p_indices)

            # Update probablity of the failure (not covering the label)
            data, indices = numba_sparse_vec_mul_ones_minus_vec(
                r_data, r_indices, p_data, p_indices
            )
            failure_prob[indices] *= data

        new_cov = 1 - np.mean(failure_prob)
        if alpha < 1:
            new_cov = alpha * new_cov + (1 - alpha) * np.asarray(calculate_tp_csr(y_proba, y_pred) / ni / k).ravel().mean()
        
        meta["utilities"].append(new_cov)
        print(
            f"  Iteration {j} finished, expected coverage: {old_cov} -> {new_cov}"
        )
        if new_cov <= old_cov + tolerance:
            break

        if filename is not None:
            save_npz(f"{filename}_pred_iter_{j}.npz", y_pred)

    meta["iters"] = j
    return y_pred, meta


def bca_coverage(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    tolerance: float = 1e-6,
    max_iter: int = 100,
    seed: int = None,
    **kwargs,
):
    """
    TODO: Add docstring
    """
    print(kwargs)
    if isinstance(y_proba, np.ndarray):
        # Invoke original dense implementation of Erik
        return bca_coverage_np(
            y_proba, k, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs
        )
    elif isinstance(y_proba, csr_matrix):
        # Invoke implementation for sparse matrices
        return bca_coverage_csr(
            y_proba, k, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs
        )
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")


# Implementations of functions for optimizing specific measures

def instance_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k):
    return np.asarray(tp / k).ravel()


def macro_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=EPS):
    return np.asarray(tp / (tp + fp + epsilon)).ravel()


def macro_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=EPS):
    return np.asarray(tp / (tp + fn + epsilon)).ravel()


def macro_fmeasure_on_conf_matrix(tp, fp, fn, tn, beta=1.0, epsilon=EPS):
    precision = macro_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
    recall = macro_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
    return (
        (1 + beta**2)
        * precision
        * recall
        / (beta**2 * precision + recall + epsilon)
    )


def block_coordinate_coverage(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 1, **kwargs
):
    return bca_coverage(y_proba, k=k, alpha=alpha, **kwargs)


def block_coordinate_macro_precision(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs
):
    return bca_with_0approx(
        y_proba, k=k, utility_func=macro_precision_on_conf_matrix, **kwargs
    )


def block_coordinate_macro_recall(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs
):
    return bca_with_0approx(
        y_proba, k=k, utility_func=macro_recall_on_conf_matrix, **kwargs
    )


def block_coordinate_macro_f1(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs
):
    return bca_with_0approx(
        y_proba, k=k, utility_func=macro_fmeasure_on_conf_matrix, **kwargs
    )


def block_coordinate_mixed_instance_prec_macro_prec(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 1, **kwargs
):
    def mixed_precision_alpha_fn(tp, fp, fn, tn):
        return (1 - alpha) * instance_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k) + alpha * macro_precision_on_conf_matrix(tp, fp, fn, tn)

    return bca_with_0approx(
        y_proba, k=k, utility_func=mixed_precision_alpha_fn, **kwargs
    )


def block_coordinate_mixed_instance_prec_macro_f1(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 1, **kwargs
):
    def mixed_precision_alpha_fn(tp, fp, fn, tn):
        return (1 - alpha) * instance_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k) + alpha * macro_fmeasure_on_conf_matrix(tp, fp, fn, tn)

    return bca_with_0approx(
        y_proba, k=k, utility_func=mixed_precision_alpha_fn, **kwargs
    )


def block_coordinate_mixed_instance_prec_macro_recall(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 1, **kwargs
):
    def mixed_precision_alpha_fn(tp, fp, fn, tn):
        return (1 - alpha) * instance_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k) + alpha * macro_recall_on_conf_matrix(tp, fp, fn, tn)

    return bca_with_0approx(
        y_proba, k=k, utility_func=mixed_precision_alpha_fn, **kwargs
    )

