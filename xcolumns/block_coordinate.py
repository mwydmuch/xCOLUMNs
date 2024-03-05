from __future__ import annotations

from time import time
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange

from .confusion_matrix import *
from .metrics import *
from .numba_csr_functions import *
from .types import *
from .utils import *
from .weighted_prediction import predict_top_k


def _get_metric_aggregation_func(metric_aggregation: str):
    if metric_aggregation == "mean":
        return np.mean
    elif metric_aggregation == "sum":
        return np.sum
    else:
        raise ValueError(
            f"Unsupported utility aggregation function: {metric_aggregation}, must be either 'mean' or 'sum'"
        )


def _get_initial_y_pred(
    y_proba: Union[np.ndarray, csr_matrix],
    init_y_pred: Union[np.ndarray, csr_matrix],
    k: int,
    random_at_k_func: Callable,
) -> Union[np.ndarray, csr_matrix]:
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
    bin_matric_func: Union[Callable, list[Callable]],
    metric_aggregation_func: Callable,
    Etp: np.ndarray,
    Efp: np.ndarray,
    Efn: np.ndarray,
    Etn: np.ndarray,
) -> np.ndarray:
    if callable(bin_matric_func):
        bin_utilities = bin_matric_func(Etp, Efp, Efn, Etn)
    else:
        bin_utilities = np.array(
            [f(Etp[i], Efp[i], Efn[i], Etn[i]) for i, f in enumerate(bin_matric_func)]
        )

    # Validate bin utilities here to omit unnecessary calculations later in _calculate_binary_gains
    if not isinstance(bin_utilities, np.ndarray):
        raise ValueError(
            f"bin_matric_func must return np.ndarray, but returned {type(bin_utilities)}"
        )

    if bin_utilities.shape != (Etp.shape[0],):
        raise ValueError(
            f"bin_matric_func must return np.ndarray of shape {Etp.shape[0]}, but returned {bin_utilities.shape}"
        )

    return metric_aggregation_func(bin_utilities)


def _calculate_binary_gains(
    bin_matric_func,
    pos_Etp: np.ndarray,
    pos_Efp: np.ndarray,
    pos_Efn: np.ndarray,
    pos_Etn: np.ndarray,
    neg_Etp: np.ndarray,
    neg_Efp: np.ndarray,
    neg_Efn: np.ndarray,
    neg_Etn: np.ndarray,
) -> np.ndarray:
    if callable(bin_matric_func):
        pos_utility = bin_matric_func(pos_Etp, pos_Efp, pos_Efn, pos_Etn)
        neg_utility = bin_matric_func(neg_Etp, neg_Efp, neg_Efn, neg_Etn)
    else:
        pos_utility = np.array(
            [
                f(pos_Etp[i], pos_Efp[i], pos_Efn[i], pos_Etn[i])
                for i, f in enumerate(bin_matric_func)
            ]
        )
        neg_utility = np.array(
            [
                f(neg_Etp[i], neg_Efp[i], neg_Efn[i], neg_Etn[i])
                for i, f in enumerate(bin_matric_func)
            ]
        )

    return pos_utility - neg_utility


def bc_with_0approx_step_np(
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    i: int,
    Etp: np.ndarray,
    Efp: np.ndarray,
    Efn: np.ndarray,
    Etn: np.ndarray,
    k: int,
    bin_matric_func: Union[Callable, list[Callable]],
    greedy: bool = False,
    maximize: bool = True,
    only_pred: bool = False,
    skip_tn: bool = False,
) -> None:
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
    pos_Etp = Etp + y_proba_i
    pos_Efp = Efp + (1 - y_proba_i)
    neg_Efn = Efn + y_proba_i
    neg_Etn = Etn

    if not skip_tn:
        neg_Etn = Etn + (1 - y_proba_i)

    gains = _calculate_binary_gains(
        bin_matric_func,
        pos_Etp / n,
        pos_Efp / n,
        Efn / n,
        Etn / n,
        Etp / n,
        Efp / n,
        neg_Efn / n,
        neg_Etn / n,
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


def bc_with_0approx_step_csr(
    y_proba: csr_matrix,
    y_pred: csr_matrix,
    i: int,
    Etp: np.ndarray,
    Efp: np.ndarray,
    Efn: np.ndarray,
    Etn: np.ndarray,
    k: int,
    bin_matric_func: Union[Callable, list[Callable]],
    greedy: bool = False,
    maximize: bool = True,
    only_pred: bool = False,
    skip_tn: bool = False,
) -> None:
    n, m = y_proba.shape
    p_start, p_end = y_pred.indptr[i], y_pred.indptr[i + 1]
    t_start, t_end = y_proba.indptr[i], y_proba.indptr[i + 1]

    p_data = y_pred.data[p_start:p_end]
    p_indices = y_pred.indices[p_start:p_end]

    t_data = y_proba.data[t_start:t_end]
    t_indices = y_proba.indices[t_start:t_end]

    # Adjust local confusion matrix
    if not greedy and not only_pred:
        numba_sub_from_unnormalized_confusion_matrix_csr(
            Etp, Efp, Efn, Etn, t_data, t_indices, p_data, p_indices, skip_tn=skip_tn
        )

    neg_Etp = Etp[t_indices]
    neg_Efp = Efp[t_indices]
    pos_Efn = Efn[t_indices]

    pos_Etpp = (neg_Etp + t_data) / n
    pos_Efpp = (neg_Efp + (1 - t_data)) / n
    neg_Efnn = (pos_Efn + t_data) / n

    neg_Etp /= n
    neg_Efp /= n
    pos_Efn /= n

    pos_Etn = Etn[t_indices]
    neg_Etnn = pos_Etn
    if not skip_tn:
        neg_Etnn = (pos_Etn + (1 - t_data)) / n
        pos_Etn /= n

    # Calculate gain and selection
    gains = _calculate_binary_gains(
        bin_matric_func,
        pos_Etpp,
        pos_Efpp,
        pos_Efn,
        pos_Etn,
        neg_Etp,
        neg_Efp,
        neg_Efnn,
        neg_Etnn,
    )
    gains = np.asarray(gains).ravel()

    if not maximize:
        gains = -gains

    # Update select labels with the best gain and update prediction
    y_pred.data, y_pred.indices, y_pred.indptr = numba_set_gains_csr(
        y_pred.data, y_pred.indices, y_pred.indptr, gains, t_indices, i, k, 0.0
    )

    # Update local confusion matrix
    if not only_pred:
        numba_add_to_unnormalized_confusion_matrix_csr(
            Etp, Efp, Efn, Etn, t_data, t_indices, p_data, p_indices, skip_tn=skip_tn
        )


def predict_using_bc_with_0approx(
    y_proba: Union[np.ndarray, csr_matrix],
    bin_matric_func: Union[Callable, list[Callable]],
    k: int,
    metric_aggregation: str = "mean",  # "mean" or "sum"
    maximize=True,
    tolerance: float = 1e-6,
    init_y_pred: Union[
        str, np.ndarray, csr_matrix
    ] = "random",  # "random", "topk", "greedy", np.ndarray or csr_matrix
    max_iter: int = 100,
    shuffle_order: bool = True,
    skip_tn=False,
    return_meta: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> Union[np.ndarray, csr_matrix]:
    """
    TODO: Add docstring

    BCA with 0-approximation here uses tp, fp, fn, tn matrics parametrization of the confussion matrix,
    as opposed to algorithms presented in the final version of the paper, which use t, q, p parametrization.
    However both algorithms are equivalent.
    """

    log_info(
        f"Starting optimization of ETU utility metric block coordinate {'ascent' if maximize else 'descent'} algorithm ...",
        verbose,
    )

    # Initialize the meta data dictionary
    meta = {"utilities": [], "iters": 0, "time": time()}

    # Get specialized functions
    if isinstance(y_proba, np.ndarray):
        bc_with_0approx_step_func = bc_with_0approx_step_np
        random_at_k_func = random_at_k_np
    elif isinstance(y_proba, csr_matrix):
        bc_with_0approx_step_func = bc_with_0approx_step_csr
        random_at_k_func = random_at_k_csr
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    n, m = y_proba.shape

    # Get aggregation function
    metric_aggregation_func = _get_metric_aggregation_func(metric_aggregation)

    # Initialize the prediction matrix
    log_info(f"  Initializing initial prediction ...", verbose)
    greedy = init_y_pred == "greedy"
    y_pred = _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func)

    # Initialize the instance order and set seed for shuffling
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    for j in range(1, max_iter + 1):
        log_info(f"  Starting iteration {j}/{max_iter} ...", verbose)

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
            log_info("    Calculating expected confusion matrix ...", verbose)
            Etp, Efp, Efn, Etn = calculate_confusion_matrix(
                y_proba, y_pred, normalize=False, skip_tn=skip_tn
            )

        old_utility = _calculate_utility(
            bin_matric_func,
            metric_aggregation_func,
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
                bin_matric_func,
                greedy=greedy,
                maximize=maximize,
                skip_tn=skip_tn,
            )

        new_utility = _calculate_utility(
            bin_matric_func,
            metric_aggregation_func,
            Etp / n,
            Efp / n,
            Efn / n,
            Etn / n,
        )

        greedy = False
        meta["iters"] = j
        meta["utilities"].append(new_utility)

        log_info(
            f"    Iteration {j}/{max_iter} finished, expected metric value: {old_utility} -> {new_utility}",
            verbose,
        )
        if abs(new_utility - old_utility) < tolerance:
            log_info(
                f"  Stopping because improvement of expected metric value is smaller than {tolerance}",
                verbose,
            )
            break

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


# Implementations of specialized BC for coverage


def bc_for_coverage_step_np(
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


def bc_for_coverage_step_csr(
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
    p_start, p_end = y_pred.indptr[i], y_pred.indptr[i + 1]
    t_start, t_end = y_proba.indptr[i], y_proba.indptr[i + 1]

    p_data = y_pred.data[p_start:p_end]
    p_indices = y_pred.indices[p_start:p_end]

    t_data = y_proba.data[t_start:t_end]
    t_indices = y_proba.indices[t_start:t_end]

    # Adjust estimates of failure probability (not covering the label)
    if not greedy:
        data, indices = numba_csr_vec_mul_ones_minus_vec(
            p_data, p_indices, t_data, t_indices
        )
        Ef[indices] /= data

    # Calculate gain and selection
    gains = Ef[t_indices] * t_data
    if alpha < 1:
        gains = alpha * gains + (1 - alpha) * t_data / k
    if gains.size > k:
        top_k = np.argpartition(-gains, k)[:k]
        y_pred.indices[p_start:p_end] = sorted(t_indices[top_k])
    else:
        t_indices = np.resize(t_indices, k)
        t_indices[gains.size :] = 0
        y_pred.indices[p_start:p_end] = sorted(t_indices)

    # Update estimates of failure probability
    data, indices = numba_csr_vec_mul_ones_minus_vec(
        p_data, p_indices, t_data, t_indices
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


def predict_optimizing_coverage_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[
        str, np.ndarray, csr_matrix
    ] = "random",  # "random", "topk", "random", or csr_matrix
    max_iter: int = 100,
    shuffle_order: bool = True,
    return_meta: bool = False,
    seed: int = None,
    verbose: bool = False,
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
        bc_coverage_step_func = bc_for_coverage_step_np
        random_at_k_func = random_at_k_np
    elif isinstance(y_proba, csr_matrix):
        bc_coverage_step_func = bc_for_coverage_step_csr
        random_at_k_func = random_at_k_csr
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    # Initialize the prediction matrix
    log_info(f"  Initializing starting prediction ...", verbose)
    greedy = init_y_pred == "greedy"
    y_pred = _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func)

    # y_proba.data = np.minimum(y_proba.data, 1 - 1e-9)

    # Initialize the instance order and set seed for shuffling
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    for j in range(1, max_iter + 1):
        log_info(f"  Starting iteration {j}/{max_iter} ...", verbose)

        if shuffle_order:
            rng.shuffle(order)

        if greedy:
            Ef = np.ones(m, dtype=FLOAT_TYPE)
        else:
            if isinstance(y_proba, np.ndarray):
                Ef = np.product(1 - y_pred * y_proba, axis=0)
            elif isinstance(y_proba, csr_matrix):
                Ef = numba_calculate_prod_0_csr_mat_mul_ones_minus_mat(
                    *unpack_csr_matrices(y_pred, y_proba), n, m
                )

        old_cov = _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha)

        for i in tqdm(order, disable=(not verbose)):
            bc_coverage_step_func(y_proba, y_pred, i, Ef, k, alpha, greedy=greedy)

        new_cov = _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha)

        greedy = False
        meta["iters"] = j
        meta["utilities"].append(new_cov)

        log_info(
            f"    Iteration {j}/{max_iter} finished, expected coverage: {old_cov} -> {new_cov}",
            verbose,
        )
        if new_cov <= old_cov + tolerance:
            log_info(
                f"  Stopping because improvement of expected coverage is smaller than {tolerance}",
                verbose,
            )
            break

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred
