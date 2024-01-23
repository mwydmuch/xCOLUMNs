from __future__ import annotations

from time import time
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange

from .default_types import *
from .metrics_on_conf_matrix import *
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


def _get_initial_y_pred(
    y_proba: Union[np.ndarray, csr_matrix],
    init_y_pred: Union[np.ndarray, csr_matrix],
    k: int,
    random_at_k_func: Callable,
):
    n, m = y_proba.shape

    if init_y_pred in ["random", "greedy"]:
        y_pred = random_at_k_func((n, m), k)
    elif init_y_pred == "topk":
        y_pred = predict_top_k(y_proba, k, return_meta=False)
    elif isinstance(init_y_pred, (np.ndarray, csr_matrix)):
        if init_y_pred.shape != (n, m):
            raise ValueError(
                f"init_y_pred must have shape (n, m) = ({n}, {m}), but has shape {init_y_pred.shape}"
            )
        y_pred = init_y_pred
    else:
        raise ValueError(
            f"init_y_pred must be ndarray, csr_matrix or str in ['random', 'greedy', 'topk'], but has type {type(init_y_pred)}"
        )
    return y_pred


def _calculate_utility(
    bin_utility_func: Union[Callable, list[Callable]],
    utility_aggregation_func: Callable,
    Etp: np.ndarray,
    Efp: np.ndarray,
    Efn: np.ndarray,
    Etn: np.ndarray,
):
    if callable(bin_utility_func):
        bin_utilities = bin_utility_func(Etp, Efp, Efn, Etn)
    else:
        bin_utilities = np.array(
            [f(Etp[i], Efp[i], Efn[i], Etn[i]) for i, f in enumerate(bin_utility_func)]
        )

    # Validate bin utilities here to omit unnecessary calculations later in _calculate_binary_gains
    if not isinstance(bin_utilities, np.ndarray):
        raise ValueError(
            f"bin_utility_func must return np.ndarray, but returned {type(bin_utilities)}"
        )
    
    if bin_utilities.shape != (Etp.shape[0],):
        raise ValueError(
            f"bin_utility_func must return np.ndarray of shape {Etp.shape[0]}, but returned {bin_utilities.shape}"
        )

    return utility_aggregation_func(bin_utilities)


def _calculate_binary_gains(
    bin_utility_func,
    p_Etp: np.ndarray,
    p_Efp: np.ndarray,
    p_Efn: np.ndarray,
    p_Etn: np.ndarray,
    n_Etp: np.ndarray,
    n_Efp: np.ndarray,
    n_Efn: np.ndarray,
    n_Etn: np.ndarray,
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
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    i: int,
    Etp: np.ndarray,
    Efp: np.ndarray,
    Efn: np.ndarray,
    Etn: np.ndarray,
    k: int,
    bin_utility_func: Union[Callable, list[Callable]],
    greedy: bool = False,
    maximize: bool = True,
    only_pred: bool = False,
    skip_tn: bool = False,
):
    n, m = y_proba.shape
    y_proba_i = y_proba[i, :]
    y_pred_i = y_pred[i, :]

    # Adjust local confusion matrix
    if not greedy and not only_pred:
        Etp -= y_pred_i * y_proba_i
        Efp -= y_pred_i * (1 - y_proba_i)
        Efn -= (1 - y_pred_i) * y_proba_i

        if not skip_tn:
            Etn -= (1 - y_pred_i) * (1 - y_proba_i)

    # Calculate gain and selection
    Etpp = Etp + y_proba_i
    Efpp = Efp + (1 - y_proba_i)
    Efnn = Efn + y_proba_i
    Etnn = Etn

    if not skip_tn:
        Etnn = Etn + (1 - y_proba_i)

    gains = _calculate_binary_gains(
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

    # Update prediction
    y_pred_i[:] = 0.0

    if k > 0:
        top_k = np.argpartition(gains, k)[:k]
        y_pred_i[top_k] = 1.0
    else: 
        y_pred_i[gains <= 0] = 1.0

    # Update local confusion matrix
    if not only_pred:
        Etp += y_pred_i * y_proba_i
        Efp += y_pred_i * (1 - y_proba_i)
        Efn += (1 - y_pred_i) * y_proba_i

        if not skip_tn:
            Etn += (1 - y_pred_i) * (1 - y_proba_i)


def bc_with_0approx_csr_step(
    y_proba: csr_matrix,
    y_pred: csr_matrix,
    i: int,
    Etp: np.ndarray,
    Efp: np.ndarray,
    Efn: np.ndarray,
    Etn: np.ndarray,
    k: int,
    bin_utility_func: Union[Callable, list[Callable]],
    greedy: bool = False,
    maximize: bool = True,
    only_pred: bool = False,
    skip_tn: bool = False,
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
        tp_data, tp_indices = numba_sparse_vec_mul_vec(
            r_data, r_indices, p_data, p_indices
        )
        Etp[tp_indices] -= tp_data
        fp_data, fp_indices = numba_sparse_vec_mul_ones_minus_vec(
            r_data, r_indices, p_data, p_indices
        )
        Efp[fp_indices] -= fp_data
        fn_data, fn_indices = numba_sparse_vec_mul_ones_minus_vec(
            p_data, p_indices, r_data, r_indices
        )
        Efn[fn_indices] -= fn_data

        if not skip_tn:
            Etn -= 1
            Etn[tp_indices] += tp_data
            Etn[fp_indices] += fp_data
            Etn[fn_indices] += fn_data

    n_Etp = Etp[p_indices]
    n_Efp = Efp[p_indices]
    p_Efn = Efn[p_indices]

    p_Etpp = (n_Etp + p_data) / n
    p_Efpp = (n_Efp + (1 - p_data)) / n
    n_Efnn = (p_Efn + p_data) / n

    n_Etp /= n
    n_Efp /= n
    p_Efn /= n

    p_Etn = Etn[p_indices]
    n_Etnn = p_Etn
    if not skip_tn:
        n_Etnn = (p_Etn + (1 - p_data)) / n
        p_Etn /= n

    # Calculate gain and selection
    gains = _calculate_binary_gains(
        bin_utility_func, p_Etpp, p_Efpp, p_Efn, p_Etn, n_Etp, n_Efp, n_Efnn, n_Etnn
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
        tp_data, tp_indices = numba_sparse_vec_mul_vec(
            r_data, r_indices, p_data, p_indices
        )
        Etp[tp_indices] += tp_data
        fp_data, fp_indices = numba_sparse_vec_mul_ones_minus_vec(
            r_data, r_indices, p_data, p_indices
        )
        Efp[fp_indices] += fp_data
        fn_data, fn_indices = numba_sparse_vec_mul_ones_minus_vec(
            p_data, p_indices, r_data, r_indices
        )
        Efn[fn_indices] += fn_data

        if not skip_tn:
            Etn += 1
            Etn[tp_indices] -= tp_data
            Etn[fp_indices] -= fp_data
            Etn[fn_indices] -= fn_data


def bc_with_0approx(
    y_proba: Union[np.ndarray, csr_matrix],
    bin_utility_func: Union[Callable, list[Callable]],
    k: int,
    utility_aggregation: str = "mean",  # "mean" or "sum"
    tolerance: float = 1e-6,
    init_y_pred: Union[
        str, np.ndarray, csr_matrix
    ] = "random",  # "random", "topk", "greedy", np.ndarray or csr_matrix
    max_iter: int = 100,
    shuffle_order: bool = True,
    maximize=True,
    skip_tn=False,
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

    # Initialize the meta data dictionary
    meta = {"utilities": [], "iters": 0, "time": time()}

    # Get specialized functions
    if isinstance(y_proba, np.ndarray):
        bc_with_0approx_step_func = bc_with_0approx_np_step
        random_at_k_func = random_at_k_np
    elif isinstance(y_proba, csr_matrix):
        bc_with_0approx_step_func = bc_with_0approx_csr_step
        random_at_k_func = random_at_k_csr
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    # y_proba is ndarray or csr_matrix
    n, m = y_proba.shape

    # Get aggregation function
    utility_aggregation_func = _get_utility_aggregation_func(utility_aggregation)

    # Initialize the prediction matrix
    print(f"  Initializing starting prediction ...")
    greedy = init_y_pred == "greedy"
    y_pred = _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func)

    # Initialize the instance order and set seed for shuffling
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    for j in range(1, max_iter + 1):
        print(f"  Starting iteration {j} ...")

        if shuffle_order:
            rng.shuffle(order)

        # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
        # In this variant they will be all np.matrix with shape (1, m)
        if greedy:
            Etp = np.zeros(m, dtype=FLOAT_TYPE)
            Efp = np.zeros(m, dtype=FLOAT_TYPE)
            Efn = np.zeros(m, dtype=FLOAT_TYPE)
            Etn = np.zeros(m, dtype=FLOAT_TYPE)
        else:
            print("  Calculating expected confusion matrix ...")
            Etp, Efp, Efn, Etn = calculate_confusion_matrix(
                y_proba, y_pred, normalize=False, skip_tn=skip_tn
            )

        old_utility = _calculate_utility(
            bin_utility_func,
            utility_aggregation_func,
            Etp / n,
            Efp / n,
            Efn / n,
            Etn / n,
        )

        for i in tqdm(order, disable=(not verbose)):
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
                skip_tn=skip_tn,
            )

        new_utility = _calculate_utility(
            bin_utility_func,
            utility_aggregation_func,
            Etp / n,
            Efp / n,
            Efn / n,
            Etn / n,
        )

        greedy = False
        meta["iters"] = j
        meta["utilities"].append(new_utility)

        print(
            f"  Iteration {j} finished, expected utility: {old_utility} -> {new_utility}"
        )
        if abs(new_utility - old_utility) < tolerance:
            print(f"  Stopping because improvement of expected utility is smaller than {tolerance}")
            break

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


# Implementations of specialized BC for coverage


def bc_coverage_np_step(
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    i: int,
    Ef: np.ndarray,
    k: int,
    alpha: float,
    greedy: bool = False,
):
    """
    Perform a single step of block coordinate for coverage
    on a single instance i using probability estimates and predictions in dense format.
    """
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


def bc_coverage_csr_step(
    y_proba: csr_matrix,
    y_pred: csr_matrix,
    i: int,
    Ef: np.ndarray,
    k: int,
    alpha: float,
    greedy: bool = False,
):
    """
    Perform a single step of block coordinate for coverage
    on a single instance i using probability estimates and predictions in sparse format.
    """
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


def _calculate_coverage_utility(
    y_proba: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    Ef: np.ndarray,
    k: int,
    alpha: float,
):
    n, m = y_proba.shape

    cov_utility = 1 - np.mean(Ef)
    if alpha < 1:
        precision_at_k = (calculate_tp(y_proba, y_pred) / n / k).sum()
        # if isinstance(y_proba, np.ndarray):
        #     precision_at_k = (np.sum(y_pred * y_proba, axis=0) / n / k).sum()
        # elif isinstance(y_proba, csr_matrix):
        #     precision_at_k = (
        #         np.asarray(calculate_tp_csr(y_proba, y_pred) / n / k).ravel().sum()
        #     )
        cov_utility = alpha * cov_utility + (1 - alpha) * precision_at_k

    return cov_utility


def bc_coverage(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[
        str, np.ndarray, csr_matrix
    ] = "random",  # "random", "topk", "random", or csr_matrix
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

    # Initialize the meta data dictionary
    meta = {"utilities": [], "iters": 0, "time": time()}

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
    print(f"  Initializing starting prediction ...")
    greedy = init_y_pred == "greedy"
    y_pred = _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func)

    # y_proba.data = np.minimum(y_proba.data, 1 - 1e-9)

    # Initialize the instance order and set seed for shuffling
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    for j in range(1, max_iter + 1):
        print(f"  Starting iteration {j} ...")

        if shuffle_order:
            rng.shuffle(order)

        if greedy:
            Ef = np.ones(m, dtype=FLOAT_TYPE)
        else:
            if isinstance(y_proba, np.ndarray):
                Ef = np.product(1 - y_pred * y_proba, axis=0)
            elif isinstance(y_proba, csr_matrix):
                Ef = numba_calculate_prod_0_sparse_mat_mul_ones_minus_mat(
                    *unpack_csr_matrices(y_pred, y_proba), n, m
                )

        old_cov = _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha)

        for i in tqdm(order, disable=(not verbose)):
            bc_coverage_step_func(y_proba, y_pred, i, Ef, k, alpha, greedy=greedy)

        new_cov = _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha)

        greedy = False
        meta["iters"] = j
        meta["utilities"].append(new_cov)

        print(f"  Iteration {j} finished, expected coverage: {old_cov} -> {new_cov}")
        if new_cov <= old_cov + tolerance:
            break

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


def bc_instance_precision_at_k(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iter: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    def instance_precision_with_specific_k(tp, fp, fn, tn):
        return bin_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k)

    return bc_with_0approx(
        y_proba,
        bin_utility_func=instance_precision_with_specific_k,
        k=k,
        utility_aggregation="sum",
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


def bc_macro_precision(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iter: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    return bc_with_0approx(
        y_proba,
        bin_utility_func=bin_precision_on_conf_matrix,
        k=k,
        utility_aggregation="mean",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


def bc_macro_recall(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iter: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    return bc_with_0approx(
        y_proba,
        bin_utility_func=bin_recall_on_conf_matrix,
        k=k,
        utility_aggregation="mean",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


def bc_macro_f1(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iter: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    return bc_with_0approx(
        y_proba,
        bin_utility_func=bin_fmeasure_on_conf_matrix,
        k=k,
        utility_aggregation="mean",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


def bc_mixed_instance_prec_macro_prec(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
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
        return (1 - alpha) * bin_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * bin_precision_on_conf_matrix(tp, fp, fn, tn) / m

    return bc_with_0approx(
        y_proba,
        bin_utility_func=mixed_utility_fn,
        k=k,
        utility_aggregation="sum",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


def bc_mixed_instance_prec_macro_f1(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
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
        return (1 - alpha) * bin_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * bin_fmeasure_on_conf_matrix(tp, fp, fn, tn) / m

    return bc_with_0approx(
        y_proba,
        k=k,
        bin_utility_func=mixed_utility_fn,
        utility_aggregation="sum",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


def bc_mixed_instance_prec_macro_recall(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
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
        return (1 - alpha) * bin_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * bin_recall_on_conf_matrix(tp, fp, fn, tn) / m

    return bc_with_0approx(
        y_proba,
        bin_utility_func=mixed_utility_fn,
        k=k,
        utility_aggregation="sum",
        skip_tn=True,
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iter=max_iter,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )
