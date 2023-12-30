from __future__ import annotations

from time import time
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm, trange

from .default_types import FLOAT_TYPE, INT_TYPE
from .numba_csr_methods import *
from .utils import *
from .weighted_prediction import predict_top_k


def _get_utility_aggregation_func(utility_aggregation: str):
    if utility_aggregation == "mean":
        return np.mean
    elif utility_aggregation == "sum":
        return np.sum
    else:
        raise ValueError(
            f"Unsupported utility aggregation function: {utility_aggregation}, must be either 'mean' or 'sum'"
        )


def _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func):
    n, m = y_proba.shape

    if init_y_pred in ["random", "greedy"]:
        y_pred = random_at_k_func((n, m), k)
    elif init_y_pred == "topk":
        y_pred = predict_top_k(y_proba, k, return_meta=False)
    else:
        y_pred = init_y_pred
        if y_pred.shape != (n, m):
            raise ValueError(
                f"init_y_pred must have shape (n, m) = ({n}, {m}), but has shape {y_pred.shape}"
            )
        if not isinstance(y_pred, type(y_proba)):
            raise ValueError(
                f"init_y_pred must have type {type(y_proba)}, but has type {type(y_pred)}"
            )
    return y_pred


def _calculate_utility(Etp, Efp, Efn, Etn, bin_utility_func, utility_aggregation_func):
    if callable(bin_utility_func):
        return utility_aggregation_func(bin_utility_func(Etp, Efp, Efn, Etn))
    else:
        return utility_aggregation_func(
            [f(Etp, Efp, Efn, Etn) for f in bin_utility_func]
        )


def _calculate_binary_utilities(
    bin_utility_func, p_Etp, p_Efp, p_Efn, p_Etn, n_Etp, n_Efp, n_Efn, n_Etn
):
    if callable(bin_utility_func):
        p_utility = bin_utility_func(p_Etp, p_Efp, p_Efn, p_Etn)
        n_utility = bin_utility_func(n_Etp, n_Efp, n_Efn, n_Etn)
    else:
        p_utility = np.array(
            [
                f(p_Etp[i], p_Efp[i], p_Efn[i], p_Etn[i])
                for i, f in enumerate(bin_utility_func)
            ]
        )
        n_utility = np.array(
            [
                f(n_Etp[i], n_Efp[i], n_Efn[i], n_Etn[i])
                for i, f in enumerate(bin_utility_func)
            ]
        )

    return p_utility - n_utility


def bc_with_0approx_np_step(
    y_proba,
    y_pred,
    i,
    Etp,
    Efp,
    Efn,
    Etn,
    k,
    bin_utility_func,
    greedy=False,
    maximize=True,
    only_pred=False,
):
    n, m = y_proba.shape
    y_proba_i = y_proba[i, :]
    y_pred_i = y_pred[i, :]

    # Adjust local confusion matrix
    if not greedy and not only_pred:
        Etp -= y_pred_i * y_proba_i
        Efp -= y_pred_i * (1 - y_proba_i)
        Efn -= (1 - y_pred_i) * y_proba_i
        Etn -= (1 - y_pred_i) * (1 - y_proba_i)

    # Calculate gain and selection
    Etpp = Etp + y_proba_i
    Efpp = Efp + (1 - y_proba_i)
    Efnn = Efn + y_proba_i
    Etnn = Etn + (1 - y_proba_i)

    gains = _calculate_binary_utilities(
        bin_utility_func,
        Etpp / n,
        Efpp / n,
        Efn / n,
        Etn / n,
        Etp / n,
        Efp / n,
        Efnn / n,
        Etnn / n,
    )

    if maximize:
        gains = -gains

    top_k = np.argpartition(gains, k)[:k]

    # Update prediction
    y_pred_i[:] = 0.0
    y_pred_i[top_k] = 1.0

    # Update local confusion matrix
    if not only_pred:
        Etp += y_pred_i * y_proba_i
        Efp += y_pred_i * (1 - y_proba_i)
        Efn += (1 - y_pred_i) * y_proba_i
        Etn += (1 - y_pred_i) * (1 - y_proba_i)


def bc_with_0approx_csr_step(
    y_proba,
    y_pred,
    i,
    Etp,
    Efp,
    Efn,
    Etn,
    k,
    bin_utility_func,
    greedy=False,
    maximize=True,
    only_pred=False,
):
    n, m = y_proba.shape
    r_start, r_end = y_pred.indptr[i], y_pred.indptr[i + 1]
    p_start, p_end = y_proba.indptr[i], y_proba.indptr[i + 1]

    r_data = y_pred.data[r_start:r_end]
    r_indices = y_pred.indices[r_start:r_end]

    p_data = y_proba.data[p_start:p_end]
    p_indices = y_proba.indices[p_start:p_end]

    # Adjust local confusion matrix
    if not greedy and not only_pred:
        Etn -= 1
        data, indices = numba_sparse_vec_mul_vec(r_data, r_indices, p_data, p_indices)
        Etp[indices] -= data
        Etn[indices] += data
        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            r_data, r_indices, p_data, p_indices
        )
        Efp[indices] -= data
        Etn[indices] += data
        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            p_data, p_indices, r_data, r_indices
        )
        Efn[indices] -= data
        Etn[indices] += data

    n_Etp = Etp[p_indices]
    n_Efp = Efp[p_indices]
    p_Efn = Efn[p_indices]
    p_Etn = Etn[p_indices]

    p_Etpp = Etp[p_indices] + p_data
    p_Efpp = Efp[p_indices] + (1 - p_data)
    n_Efnn = Efn[p_indices] + p_data
    n_Etnn = Etn[p_indices] + (1 - p_data)

    # Calculate gain and selection
    gains = _calculate_binary_utilities(
        bin_utility_func,
        p_Etpp / n,
        p_Efpp / n,
        p_Efn / n,
        p_Etn / n,
        n_Etp / n,
        n_Efp / n,
        n_Efnn / n,
        n_Etnn / n,
    )
    gains = np.asarray(gains).ravel()

    if maximize:
        gains = -gains

    # Update select labels with the best gain and update prediction
    if gains.size > k:
        top_k = np.argpartition(gains, k)[:k]
        y_pred.indices[r_start:r_end] = sorted(p_indices[top_k])
    else:
        p_indices = np.resize(p_indices, k)
        p_indices[gains.size :] = 0
        y_pred.indices[r_start:r_end] = sorted(p_indices)

    # Update local confusion matrix
    if not only_pred:
        Etn += 1
        data, indices = numba_sparse_vec_mul_vec(r_data, r_indices, p_data, p_indices)
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


def bc_with_0approx(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    bin_utility_func: Callable | list[Callable],
    utility_aggregation: str = "mean",  # "mean" or "sum"
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",  # "random", "topk", "greedy", np.ndarray or csr_matrix
    max_iter: int = 100,
    shuffle_order: bool = True,
    maximize=True,
    seed: int = None,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    """
    TODO: Add docstring

    BCA with 0-approximation here uses tp, fp, fn, tn matrics parametrization of the confussion matrix,
    as opposed to algorithms presented in the final version of the paper, which use t, q, p parametrization.
    However both algorithms are equivalent.
    """

    n, m = y_proba.shape

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Get specialized functions
    if isinstance(y_proba, np.ndarray):
        bc_with_0approx_step_func = bc_with_0approx_np_step
        random_at_k_func = random_at_k_np
    elif isinstance(y_proba, csr_matrix):
        bc_with_0approx_step_func = bc_with_0approx_csr_step
        random_at_k_func = random_at_k_csr
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    # Get aggregation function
    utility_aggregation_func = _get_utility_aggregation_func(utility_aggregation)

    # Initialize the prediction matrix
    greedy = init_y_pred == "greedy"
    y_pred = _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func)

    # Initialize the meta data dictionary
    meta = {"utilities": [], "iters": 0, "time": time()}

    order = np.arange(n)
    for j in range(1, max_iter + 1):
        if shuffle_order:
            np.random.shuffle(order)

        # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
        # In this variant they will be all np.matrix with shape (1, m)
        if greedy:
            Etp = np.zeros(m, dtype=FLOAT_TYPE)
            Efp = np.zeros(m, dtype=FLOAT_TYPE)
            Efn = np.zeros(m, dtype=FLOAT_TYPE)
            Etn = np.zeros(m, dtype=FLOAT_TYPE)
        else:
            Etp, Efp, Efn, Etn = calculate_confusion_matrix(
                y_proba, y_pred, normalize=False
            )

        old_utility = _calculate_utility(
            Etp / n,
            Efp / n,
            Efn / n,
            Etn / n,
            bin_utility_func,
            utility_aggregation_func,
        )

        for i in tqdm(order, disable=verbose):
            bc_with_0approx_step_func(
                y_proba,
                y_pred,
                i,
                Etp,
                Efp,
                Efn,
                Etn,
                k,
                bin_utility_func,
                greedy=greedy,
                maximize=maximize,
            )

        new_utility = _calculate_utility(
            Etp / n,
            Efp / n,
            Efn / n,
            Etn / n,
            bin_utility_func,
            utility_aggregation_func,
        )

        meta["utilities"].append(new_utility)
        print(
            f"  Iteration {j} finished, expected score: {old_utility} -> {new_utility}"
        )
        if new_utility <= old_utility + tolerance:
            break

        greedy = False
        meta["iters"] = j

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


# Implementations of specialized BC for coverage


def bc_coverage_np_step(y_proba, y_pred, i, Ef, k, alpha, greedy=False):
    # Adjust estimates of probability of the failure (not covering the label)
    if greedy:
        Ef /= 1 - y_pred[i] * y_proba[i]

    # Calculate gain and selection
    gains = Ef * y_proba[i]
    if alpha < 1:
        gains = alpha * gains + (1 - alpha) * y_proba[i] / k
    top_k = np.argpartition(-gains, k)[:k]
    y_pred[i, :] = 0.0
    y_pred[i, top_k] = 1.0

    # Update estimates of failure probability
    Ef *= 1 - y_pred[i] * y_proba[i]


def bc_coverage_csr_step(y_proba, y_pred, i, Ef, k, alpha, greedy=False):
    r_start, r_end = y_pred.indptr[i], y_pred.indptr[i + 1]
    p_start, p_end = y_proba.indptr[i], y_proba.indptr[i + 1]

    r_data = y_pred.data[r_start:r_end]
    r_indices = y_pred.indices[r_start:r_end]

    p_data = y_proba.data[p_start:p_end]
    p_indices = y_proba.indices[p_start:p_end]

    # Adjust estimates of failure probability (not covering the label)
    if not greedy:
        data, indices = numba_sparse_vec_mul_ones_minus_vec(
            r_data, r_indices, p_data, p_indices
        )
        Ef[indices] /= data

    # Calculate gain and selection
    gains = Ef[p_indices] * p_data
    if alpha < 1:
        gains = alpha * gains + (1 - alpha) * p_data / k
    if gains.size > k:
        top_k = np.argpartition(-gains, k)[:k]
        y_pred.indices[r_start:r_end] = sorted(p_indices[top_k])
    else:
        p_indices = np.resize(p_indices, k)
        p_indices[gains.size :] = 0
        y_pred.indices[r_start:r_end] = sorted(p_indices)

    # Update estimates of failure probability
    data, indices = numba_sparse_vec_mul_ones_minus_vec(
        r_data, r_indices, p_data, p_indices
    )
    Ef[indices] *= data


def _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha):
    n, m = y_proba.shape

    cov_utility = 1 - np.mean(Ef)
    if alpha < 1:
        if isinstance(y_proba, np.ndarray):
            precision_at_k = (np.sum(y_pred * y_proba, axis=0) / n / k).sum()
        elif isinstance(y_proba, csr_matrix):
            precision_at_k = (
                np.asarray(calculate_tp_csr(y_proba, y_pred) / n / k).ravel().sum()
            )
        cov_utility = alpha * cov_utility + (1 - alpha) * precision_at_k

    return cov_utility


def bc_coverage(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",  # "random", "topk", "random", or csr_matrix
    max_iter: int = 100,
    shuffle_order: bool = True,
    seed: int = None,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    """
    An efficient implementation of the block coordinate-descent for coverage
    """
    n, m = y_proba.shape

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Get specialized functions
    if isinstance(y_proba, np.ndarray):
        bc_coverage_step_func = bc_coverage_np_step
        random_at_k_func = random_at_k_np
    elif isinstance(y_proba, csr_matrix):
        bc_coverage_step_func = bc_coverage_csr_step
        random_at_k_func = random_at_k_csr
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    # Initialize the prediction matrix
    greedy = init_y_pred == "greedy"
    y_pred = _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func)

    # Initialize the meta data dictionary
    meta = {"utilities": [], "iters": 0, "time": time()}

    y_proba.data = np.minimum(y_proba.data, 1 - 1e-9)  # TODO: Improve

    order = np.arange(n)
    for j in range(1, max_iter + 1):
        if shuffle_order:
            np.random.shuffle(order)

        if greedy:
            Ef = np.ones(m, dtype=FLOAT_TYPE)
        else:
            if isinstance(y_proba, np.ndarray):
                Ef = np.product(1 - y_pred * y_proba, axis=0)
            elif isinstance(y_proba, csr_matrix):
                Ef = numba_calculate_prod_1_sparse_mat_mul_ones_minus_mat(
                    *unpack_csr_matrices(y_pred, y_proba), n, m
                )

        old_cov = _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha)

        for i in tqdm(order, disable=verbose):
            bc_coverage_step_func(y_proba, y_pred, i, Ef, k, alpha, greedy=greedy)

        new_cov = _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha)

        meta["utilities"].append(new_cov)
        print(f"  Iteration {j} finished, expected coverage: {old_cov} -> {new_cov}")
        if new_cov <= old_cov + tolerance:
            break

        greedy = False
        meta["iters"] = j

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


# Implementations of binary utilities defined on the confusion matrix


def instance_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k):
    return np.asarray(tp / k).ravel()


def macro_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=1e-9):
    return np.asarray(tp / (tp + fp + epsilon)).ravel()


def macro_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=1e-9):
    return np.asarray(tp / (tp + fn + epsilon)).ravel()


def macro_fmeasure_on_conf_matrix(tp, fp, fn, tn, beta=1.0, epsilon=1e-9):
    precision = macro_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
    recall = macro_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
    return (
        (1 + beta**2)
        * precision
        * recall
        / (beta**2 * precision + recall + epsilon)
    )


# Implementations of functions for optimizing specific measures


def bc_instance_precision_at_k(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs
):
    def instance_precision_with_k_fn(tp, fp, fn, tn):
        return instance_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k)

    return bc_with_0approx(
        y_proba,
        k=k,
        bin_utility_func=instance_precision_with_k_fn,
        utility_aggregation="sum",
        **kwargs,
    )


def bc_macro_precision(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return bc_with_0approx(
        y_proba, k=k, bin_utility_func=macro_precision_on_conf_matrix, **kwargs
    )


def bc_macro_recall(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return bc_with_0approx(
        y_proba, k=k, bin_utility_func=macro_recall_on_conf_matrix, **kwargs
    )


def bc_macro_f1(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return bc_with_0approx(
        y_proba, k=k, bin_utility_func=macro_fmeasure_on_conf_matrix, **kwargs
    )


def bc_mixed_instance_prec_macro_prec(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 1, **kwargs
):
    n, m = y_proba.shape

    def mixed_precision_alpha_fn(tp, fp, fn, tn):
        return (1 - alpha) * instance_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * macro_precision_on_conf_matrix(tp, fp, fn, tn) / m

    return bc_with_0approx(
        y_proba,
        k=k,
        bin_utility_func=mixed_precision_alpha_fn,
        utility_aggregation="sum",
        **kwargs,
    )


def bc_mixed_instance_prec_macro_f1(
    y_proba: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 1, **kwargs
):
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn):
        return (1 - alpha) * instance_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * macro_fmeasure_on_conf_matrix(tp, fp, fn, tn) / m

    return bc_with_0approx(
        y_proba,
        k=k,
        bin_utility_func=mixed_utility_fn,
        utility_aggregation="sum",
        **kwargs,
    )


def bc_mixed_instance_prec_macro_recall(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int = 5,
    alpha: float = 1,
    greedy_start=False,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iter: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn):
        return (1 - alpha) * instance_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * macro_recall_on_conf_matrix(tp, fp, fn, tn) / m

    return bc_with_0approx(
        y_proba,
        k=k,
        bin_utility_func=mixed_utility_fn,
        utility_aggregation="sum",
        greedy_start=greedy_start,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )
