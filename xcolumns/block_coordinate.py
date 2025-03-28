from time import time
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange

from .confusion_matrix import *
from .metrics import *
from .numba_csr_functions import (
    numba_add_to_unnormalized_confusion_matrix_csr,
    numba_calculate_prod_csr_mat_mul_ones_minus_mat,
    numba_csr_vec_mul_ones_minus_vec,
    numba_set_gains_csr,
    numba_sub_from_unnormalized_confusion_matrix_csr,
)
from .types import TORCH_AVAILABLE, DenseMatrix, Matrix
from .utils import log_info, log_warning, ones_like, unpack_csr_matrices, zeros_like
from .weighted_prediction import predict_top_k


########################################################################################
# General block coordinate descent/ascent with 0-th order approximation
########################################################################################


def _get_initial_y_pred(
    y_proba: Matrix,
    init_y_pred: Union[str, Matrix],  # "random", "topk", "greedy", Matrix
    k: int,
    random_at_k_func: Callable,
    seed: Optional[int] = None,
) -> Matrix:
    n, m = y_proba.shape

    if isinstance(init_y_pred, str) and init_y_pred in ["random", "greedy"]:
        y_pred = random_at_k_func((n, m), k, dtype=y_proba.dtype, seed=seed)
    elif isinstance(init_y_pred, str) and init_y_pred == "top":
        y_pred = predict_top_k(y_proba, k, return_meta=False)
    elif isinstance(init_y_pred, Matrix):
        if init_y_pred.shape != (n, m):
            raise ValueError(
                f"init_y_pred must have shape (n, m) = ({n}, {m}), but has shape {init_y_pred.shape}"
            )
        y_pred = init_y_pred
    else:
        raise ValueError(
            f"init_y_pred must be np.ndarray, Torch.tensor, csr_matrix or str in ['random', 'greedy', 'top'], but has type {type(init_y_pred)}"
        )
    return y_pred


def _calculate_utility(
    binary_metric_func: Union[Callable, List[Callable]],
    metric_aggregation: str,
    Etp: DenseMatrix,
    Efp: DenseMatrix,
    Efn: DenseMatrix,
    Etn: DenseMatrix,
) -> float:
    if callable(binary_metric_func):
        binary_metric_values = binary_metric_func(Etp, Efp, Efn, Etn)
    else:
        binary_metric_values = np.array(
            [
                f(Etp[i], Efp[i], Efn[i], Etn[i])
                for i, f in enumerate(binary_metric_func)
            ]
        )

    # Validate bin utilities here to omit unnecessary calculations later in _calculate_binary_gains
    if not isinstance(binary_metric_values, np.ndarray):
        raise ValueError(
            f"binary_metric_func must return np.ndarray, but returned {type(binary_metric_values)}"
        )

    if binary_metric_values.shape != (Etp.shape[0],):
        raise ValueError(
            f"binary_metric_func must return np.ndarray of shape {Etp.shape[0]}, but returned {binary_metric_values.shape}"
        )

    if metric_aggregation == "sum":
        return binary_metric_values.sum()
    elif metric_aggregation == "mean":
        return binary_metric_values.mean()
    else:
        raise ValueError(
            f"Unsupported utility aggregation function: {metric_aggregation}, must be either 'mean' or 'sum'"
        )


def _calculate_binary_gains(
    binary_metric_func,
    pos_Etp: DenseMatrix,
    pos_Efp: DenseMatrix,
    pos_Efn: DenseMatrix,
    pos_Etn: DenseMatrix,
    neg_Etp: DenseMatrix,
    neg_Efp: DenseMatrix,
    neg_Efn: DenseMatrix,
    neg_Etn: DenseMatrix,
    metric_kwargs: Optional[Dict[str, Any]] = None,
) -> DenseMatrix:
    if metric_kwargs is None:
        metric_kwargs = {}

    if callable(binary_metric_func):
        pos_utility = binary_metric_func(
            pos_Etp, pos_Efp, pos_Efn, pos_Etn, **metric_kwargs
        )
        neg_utility = binary_metric_func(
            neg_Etp, neg_Efp, neg_Efn, neg_Etn, **metric_kwargs
        )
    else:
        pos_utility = np.array(
            [
                f(pos_Etp[i], pos_Efp[i], pos_Efn[i], pos_Etn[i], **metric_kwargs)
                for i, f in enumerate(binary_metric_func)
            ]
        )
        neg_utility = np.array(
            [
                f(neg_Etp[i], neg_Efp[i], neg_Efn[i], neg_Etn[i], **metric_kwargs)
                for i, f in enumerate(binary_metric_func)
            ]
        )

    return pos_utility - neg_utility


def _bc_with_0approx_step_dense(
    y_proba: DenseMatrix,
    y_pred: DenseMatrix,
    i: int,
    Etp: DenseMatrix,
    Efp: DenseMatrix,
    Efn: DenseMatrix,
    Etn: DenseMatrix,
    k: int,
    binary_metric_func: Union[Callable, List[Callable]],
    normalize_conf_matrix: bool = True,
    metric_kwargs: Optional[Dict[str, Any]] = None,
    greedy: bool = False,
    maximize: bool = True,
    only_pred: bool = False,
    skip_tn: bool = False,
) -> None:
    n = 1
    if normalize_conf_matrix:
        n, _ = y_proba.shape

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
        binary_metric_func,
        pos_Etp / n,
        pos_Efp / n,
        Efn / n,
        Etn / n,
        Etp / n,
        Efp / n,
        neg_Efn / n,
        neg_Etn / n,
        metric_kwargs=metric_kwargs,
    )

    if maximize:
        gains = -gains

    # Update prediction
    y_pred_i[:] = 0.0

    if k > 0:
        if isinstance(gains, np.ndarray):
            top_k = np.argpartition(gains, k)[:k]
        elif TORCH_AVAILABLE and isinstance(gains, torch.Tensor):
            _, top_k = torch.topk(gains, k)
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


def _bc_with_0approx_step_csr(
    y_proba: csr_matrix,
    y_pred: csr_matrix,
    i: int,
    Etp: np.ndarray,
    Efp: np.ndarray,
    Efn: np.ndarray,
    Etn: np.ndarray,
    k: int,
    binary_metric_func: Union[Callable, List[Callable]],
    normalize_conf_matrix: bool = True,
    metric_kwargs: Optional[Dict[str, Any]] = None,
    greedy: bool = False,
    maximize: bool = True,
    only_pred: bool = False,
    skip_tn: bool = False,
) -> None:
    n = 1
    if normalize_conf_matrix:
        n, _ = y_proba.shape

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
        binary_metric_func,
        pos_Etpp,
        pos_Efpp,
        pos_Efn,
        pos_Etn,
        neg_Etp,
        neg_Efp,
        neg_Efnn,
        neg_Etnn,
        metric_kwargs=metric_kwargs,
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
    y_proba: Matrix,
    binary_metric_func: Union[Callable, List[Callable]],
    k: int,
    metric_aggregation: str = "mean",  # "mean" or "sum"
    normalize_conf_matrix: bool = True,
    metric_kwargs: Optional[Dict[str, Any]] = None,
    maximize: bool = True,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, Matrix] = "top",  # "random", "top", "greedy", Matrix
    max_iters: int = 100,
    shuffle_order: bool = True,
    skip_tn: bool = False,
    return_meta: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> Union[Matrix, Tuple[Matrix, Dict[str, Any]]]:
    r"""
    Predicts for each instance (row) in a provided
    matrix of conditional probabilities estimates of labels :math:`\boldsymbol{H}`, (**y_proba**),
    where each element :math:`\eta_{ij} = P(y_j|x_i)`
    is the probability of the label :math:`j` for the instance :math:`i`.
    It uses block coordinate ascent/descent with 0-th order approximation of Expected Test Utlity (ETU) objective
    optimizing a given metric, that decomposes into a sum/mean of binary metrics for each label:

    .. math::
        \text{metric}(\boldsymbol{C}) = \sum_{j=1}^{m} \text{or} \prod_{j=1}^{m} \text{binary_metric}(\text{TP}_j, \text{FP}_j, \text{FN}_j, \text{TN}_j)

    The algorithm iterate over instances. It changes the predictions for the one that optimizes the metric the most for one instance at a time.
    The algorithm stops when the improvement of the metric is smaller than the given **tolerance** or the maximum number of iterations **max_iters** is reached.

    Args:
        y_proba: The matrix of predicted probabilities of shape (n, m).
        binary_metric_func: The binary metric function or a list of binary metric functions.
                            It needs to take four arguments that are vectors of:
                            True Positives, False Positives, False Negatives, True Negatives for each label, and return a vector of metric values for each label.
                            (``binary_metric_func(tp, fp, fn, tn)``).
                            If a list of functions is provided, the metric is calculated as a sum of the metrics calculated by each function.
        k: The budget of positive labels per instance. If **k** is 0, the algorithm optimizes for the metric without any budget constraint.
        metric_aggregation: The aggregation function for the binary metric(s), that forms the final metrics (objective). Either "mean" or "sum". It does not affect the optimization, only the value of the returned metric.
        normalize_conf_matrix: Whether to normalize the confusion matrix before calculating the metric.
        metric_kwargs: Additional keyword arguments for the binary metric function.
        maximize: Whether to maximize the metric.
        tolerance: Defines the stopping condition, if the expected improvement of the metric is smaller than **tolerance** the algorithm stops.
        init_y_pred: The initial prediction matrix. It can be either "random", "top", "greedy" or a matrix of shape (n, m).
        max_iters: The maximum number of iterations.
        shuffle_order: Whether to shuffle the order of instances in each iteration.
        skip_tn: Whether to skip the calculation of True Negatives in the confusion matrix, if the metric does not use the True Negatives, this can speed up the calculation, especially when using sparse matrices.
        return_meta: Whether to return the meta information.
        seed: The seed for the random number generator.
        verbose: Whether to print the progress.

    Returns:
        The predicted labels matrix of shape (n, m): the shape and type of the matrix is the same as **y_proba**. If **return_meta** is True, additionally, a dictionary is returned, that contains the time taken to calculate the prediction, the number of iterations, and expected performance at each iteration.

    Example:
        Example of maximizng macro-averaged F1 score using block coordinate ascent algorithm:

        .. code-block:: python

            from xcolumns.block_coordinate import predict_using_bc_with_0approx
            y_proba = some_model.predict_proba(X_test) # Marginal probabilities of labels, matrix of shape (n, m)

            # Define the binary metric function to optimize (eg. F1-score)
            def my_binary_f1_score_on_conf_matrix(tp, fp, fn, tn):
                return 2 * tp / (2 * tp + fp + fn + 1e-9)

            y_pred = predict_using_bc_with_0approx(
                y_proba,
                my_binary_f1_score_on_conf_matrix,
                k=3,                                # The udget of positive labels per instance
                metric_aggregation='mean',          # The aggregation function for the binary metric, here mean results in macro-averaged F1-score
                maximize=True                       # Maximize the metric
            ) # Returns the predicted labels matrix of shape (n, m)
    """

    log_info(
        f"Starting optimization of ETU metric using block coordinate {'ascent (maximization)' if maximize else 'descent (minimization)'} algorithm ...",
        verbose,
    )
    if k > 0:
        log_info(f"  Budget k: {k}", verbose)
    log_info(
        f"  Tolerance (stopping condition): {tolerance}, max iterations: {max_iters}",
        verbose,
    )

    # Initialize the meta data dictionary
    meta = {"utilities": [], "iters": 0, "time": time()}

    # Check k and th
    if not isinstance(k, int):
        raise ValueError("k must be an integer")

    # Get specialized functions
    if isinstance(y_proba, DenseMatrix):
        bc_with_0approx_step_func = _bc_with_0approx_step_dense
        random_at_k_func = random_at_k_np
    elif isinstance(y_proba, csr_matrix):
        bc_with_0approx_step_func = _bc_with_0approx_step_csr
        random_at_k_func = random_at_k_csr
    else:
        raise ValueError(
            "y_proba must be either np.ndarray, torch.Tensor, or csr_matrix"
        )

    n, m = y_proba.shape
    if not normalize_conf_matrix:
        n = 1

    # Initialize the prediction matrix
    log_info(f"  Initializing initial prediction ...", verbose)
    greedy = isinstance(init_y_pred, str) and init_y_pred == "greedy"
    y_pred = _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func, seed=seed)

    # Initialize the instance order and set seed for shuffling
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    for j in range(1, max_iters + 1):
        log_info(f"  Starting iteration {j}/{max_iters} ...", verbose)

        if shuffle_order:
            rng.shuffle(order)

        # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
        # In this variant they will be all np.matrix with shape (1, m)
        if greedy:
            Etp = zeros_like(y_proba, shape=(m,))
            Efp = zeros_like(y_proba, shape=(m,))
            Efn = zeros_like(y_proba, shape=(m,))
            Etn = zeros_like(y_proba, shape=(m,))
        else:
            log_info("    Calculating expected confusion matrix ...", verbose)
            Etp, Efp, Efn, Etn = calculate_confusion_matrix(
                y_proba, y_pred, normalize=False, skip_tn=skip_tn
            )

        old_utility = _calculate_utility(
            binary_metric_func,
            metric_aggregation,
            Etp / n,
            Efp / n,
            Efn / n,
            Etn / n,
        )

        for i in order:
            bc_with_0approx_step_func(
                y_proba,
                y_pred,
                i,
                Etp,
                Efp,
                Efn,
                Etn,
                k,
                binary_metric_func,
                metric_kwargs=metric_kwargs,
                greedy=greedy,
                maximize=maximize,
                skip_tn=skip_tn,
            )

        new_utility = _calculate_utility(
            binary_metric_func,
            metric_aggregation,
            Etp / n,
            Efp / n,
            Efn / n,
            Etn / n,
        )

        greedy = False
        meta["iters"] = j
        meta["utilities"].append(new_utility)

        log_info(
            f"    Iteration {j}/{max_iters} finished, expected metric value: {old_utility} -> {new_utility}",
            verbose,
        )
        if (maximize and new_utility - old_utility < tolerance) or (
            not maximize and new_utility - old_utility > tolerance
        ):  # abs is used to handle maximization/minimization cases
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


########################################################################################
# Implementations of specialized BCA for coverage at k
########################################################################################


def _bc_for_coverage_step_dense(
    y_proba: DenseMatrix,
    y_pred: DenseMatrix,
    i: int,
    Ef: DenseMatrix,
    k: int,
    alpha: float,
    greedy: bool = False,
) -> None:
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
    if isinstance(gains, np.ndarray):
        top_k = np.argpartition(-gains, k)[:k]
    elif TORCH_AVAILABLE and isinstance(gains, torch.Tensor):
        _, top_k = torch.topk(gains, k)
    y_pred[i, :] = 0.0
    y_pred[i, top_k] = 1.0

    # Update estimates of failure probability
    Ef *= 1 - y_pred[i] * y_proba[i]


def _bc_for_coverage_step_csr(
    y_proba: csr_matrix,
    y_pred: csr_matrix,
    i: int,
    Ef: np.ndarray,
    k: int,
    alpha: float,
    greedy: bool = False,
) -> None:
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
        Ef[indices] /= data + 1e-12

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
    y_proba: Matrix,
    y_pred: Matrix,
    Ef: DenseMatrix,
    k: int,
    alpha: float,
) -> float:
    n, m = y_proba.shape

    cov_utility = 1 - Ef.mean()
    if alpha < 1:
        precision_at_k = (calculate_tp(y_proba, y_pred) / n / k).sum()
        cov_utility = alpha * cov_utility + (1 - alpha) * precision_at_k

    return cov_utility


def predict_optimizing_coverage_using_bc(
    y_proba: Matrix,
    k: int,
    alpha: float = 1,
    tolerance: float = 1e-6,
    init_y_pred: Union[
        str, np.ndarray, csr_matrix
    ] = "random",  # "random", "topk", "random", or csr_matrix
    max_iters: int = 100,
    shuffle_order: bool = True,
    return_meta: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Union[Matrix, Tuple[Matrix, Dict[str, Any]]]:
    """
    An efficient implementation of the block coordinate-descent for coverage.

    TODO: Add more details
    """
    log_info(
        f"Starting optimization of ETU coverage@{k} metric using block coordinate ascent algorithm ...",
        verbose,
    )

    # Check k
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be an integer > 0")

    n, m = y_proba.shape

    # Initialize the meta data dictionary
    meta = {"utilities": [], "iters": 0, "time": time()}

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Get specialized functions
    if isinstance(y_proba, DenseMatrix):
        bc_coverage_step_func = _bc_for_coverage_step_dense
        random_at_k_func = random_at_k_np
    elif isinstance(y_proba, csr_matrix):
        bc_coverage_step_func = _bc_for_coverage_step_csr
        random_at_k_func = random_at_k_csr
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    # Initialize the prediction matrix
    log_info(f"  Initializing starting prediction ...", verbose)
    greedy = isinstance(init_y_pred, str) and init_y_pred == "greedy"
    y_pred = _get_initial_y_pred(y_proba, init_y_pred, k, random_at_k_func)

    # y_proba.data = np.minimum(y_proba.data, 1 - 1e-9)

    # Initialize the instance order and set seed for shuffling
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    for j in range(1, max_iters + 1):
        log_info(f"  Starting iteration {j}/{max_iters} ...", verbose)

        if shuffle_order:
            rng.shuffle(order)

        if greedy:
            Ef = ones_like(y_pred, shape=(m,))  # np.ones(m, dtype=FLOAT_TYPE)
        else:
            if isinstance(y_proba, np.ndarray):
                Ef = np.product(1 - y_pred * y_proba, axis=0)
            elif isinstance(y_proba, csr_matrix):
                Ef = numba_calculate_prod_csr_mat_mul_ones_minus_mat(
                    *unpack_csr_matrices(y_pred, y_proba), n, m
                )

        old_cov = _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha)

        for i in order:
            bc_coverage_step_func(y_proba, y_pred, i, Ef, k, alpha, greedy=greedy)

        new_cov = _calculate_coverage_utility(y_proba, y_pred, Ef, k, alpha)

        greedy = False
        meta["iters"] = j
        meta["utilities"].append(new_cov)

        log_info(
            f"    Iteration {j}/{max_iters} finished, expected coverage: {old_cov} -> {new_cov}",
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


########################################################################################
# Wrapper functions for BC algorithm for specific metrics
########################################################################################


def make_bc_wrapper(
    binary_metric_func: Callable,
    metric_name: str,
    maximize: bool = True,
    metric_aggregation: str = "mean",
    skip_tn: bool = False,
    warn_k_eq_0: bool = False,
):
    """
    Factory function that creates a wrapper function for predicting for a given test set
    optimizing a given metric using using block coordinate ascent/descent method (:func:`predict_using_bc_with_0approx`).

    Args:
        metric_func: The metric function to optimize.
        metric_name: The name of the metric that will be used in docstring.
        maximize: Whether to maximize the metric.
        metric_aggregation: The aggregation function for the metric. Either "mean" or "sum".
        skip_tn: Whether to skip the calculation of True Negatives in the confusion matrix.
        warn_k_eq_0: Whether to warn if the budget **k** equal to 0 leads to degenerated solution.

    Returns:
        The wrapper function.
    """

    def predict_optimizing_metric_using_bc(y_proba: Matrix, k: int, **kwargs):
        if warn_k_eq_0 and k == 0:
            log_warning(
                f"Warning: k=0 results in degenerated solution for {metric_name}!",
            )

        return predict_using_bc_with_0approx(
            y_proba,
            binary_metric_func,
            k,
            metric_aggregation=metric_aggregation,
            maximize=maximize,
            skip_tn=skip_tn,
            **kwargs,
        )

    predict_optimizing_metric_using_bc.__doc__ = f"""
    Find a randomized classifier that maximizes {metric_name} metric using Frank-Wolfe algorithm.
    It is equivalent to calling ``find_classifier_using_fw(y_true, y_proba, {binary_metric_func.__name__}, k, ..., metric_aggregation={metric_aggregation}, maximize={maximize}, skip_tn={skip_tn})`` function.
    See :func:`predict_using_bc_with_0approx` for more details and a description of arguments.
    """

    return add_kwargs_to_signature(
        predict_optimizing_metric_using_bc,
        predict_using_bc_with_0approx,
        skip=["metric_func", "metric_aggregation", "maximize", "skip_tn"],
    )


predict_optimizing_macro_precision_using_bc = make_bc_wrapper(
    binary_precision_on_conf_matrix,
    "macro-averaged precision",
    metric_aggregation="mean",
    maximize=True,
    skip_tn=True,
    warn_k_eq_0=True,
)
predict_optimizing_macro_recall_using_bc = make_bc_wrapper(
    binary_recall_on_conf_matrix,
    "macro-averaged recall",
    metric_aggregation="mean",
    maximize=True,
    skip_tn=True,
    warn_k_eq_0=True,
)
predict_optimizing_macro_f1_score_using_bc = make_bc_wrapper(
    binary_f1_score_on_conf_matrix,
    "macro-averaged F1 score",
    metric_aggregation="mean",
    maximize=True,
    skip_tn=True,
)
predict_optimizing_macro_jaccard_score_using_bc = make_bc_wrapper(
    binary_jaccard_score_on_conf_matrix,
    "macro-averaged Jaccard score",
    maximize=True,
    skip_tn=True,
)
predict_optimizing_macro_balanced_accuracy_using_bc = make_bc_wrapper(
    binary_balanced_accuracy_on_conf_matrix,
    "macro-averaged balanced accuracy",
    maximize=True,
)
predict_optimizing_macro_hmean_using_bc = make_bc_wrapper(
    binary_hmean_on_conf_matrix, "macro-averaged H-mean", maximize=True
)
predict_optimizing_macro_gmean_using_bc = make_bc_wrapper(
    binary_gmean_on_conf_matrix, "macro-averaged G-mean", maximize=True
)


def predict_optimizing_instance_precision_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    tolerance: float = 1e-6,
    init_y_pred: Union[str, np.ndarray, csr_matrix] = "random",
    max_iters: int = 100,
    shuffle_order: bool = True,
    verbose: bool = False,
    return_meta: bool = False,
    **kwargs,
):
    """
    This function is a wrapper for using block coordinate ascent with instance precision as the target metric.
    See :func:`predict_using_bc_with_0approx` for more details and a description of parameters.
    """

    def instance_precision_with_specific_k(tp, fp, fn, tn):
        return binary_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k)

    return predict_using_bc_with_0approx(
        y_proba,
        binary_metric_func=instance_precision_with_specific_k,
        k=k,
        metric_aggregation="sum",
        tolerance=tolerance,
        init_y_pred=init_y_pred,
        max_iters=max_iters,
        shuffle_order=shuffle_order,
        verbose=verbose,
        return_meta=return_meta,
        **kwargs,
    )


########################################################################################
# Wrapper functions for BC algorithm for mixed metrics
########################################################################################

# def binary_mixed_instance_precision_and_macro_precision(tp, fp, fn, tn, alpha=0.5, epsilon=1e-9):
#     return (1 - alpha) * binary_precision_at_k_on_conf_matrix(
#         tp, fp, fn, tn, k
#     ) + alpha * binary_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon) / m


def predict_optimizing_mixed_instance_precision_and_macro_precision_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    **kwargs,
):
    """
    This function is a wrapper for using block coordinate ascent
    with metric being a weighted average of instance precision and macro-averaged precision as the target metric.
    See :func:`predict_using_bc_with_0approx` for more details and a description of parameters.
    """
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn, epsilon=1e-9):
        return (1 - alpha) * binary_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * binary_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon) / m

    return predict_using_bc_with_0approx(
        y_proba,
        binary_metric_func=mixed_utility_fn,
        k=k,
        metric_aggregation="sum",
        skip_tn=True,
        **kwargs,
    )


def predict_optimizing_mixed_instance_precision_and_macro_recall_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    **kwargs,
):
    """
    This function is a wrapper for using block coordinate ascent
    with metric being a weighted average of instance precision and macro-averaged recall as the target metric.
    See :func:`predict_using_bc_with_0approx` for more details and a description of parameters.
    """
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn, epsilon=1e-9):
        return (1 - alpha) * binary_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * binary_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon) / m

    return predict_using_bc_with_0approx(
        y_proba,
        binary_metric_func=mixed_utility_fn,
        k=k,
        metric_aggregation="sum",
        skip_tn=True,
        **kwargs,
    )


def predict_optimizing_mixed_instance_precision_and_macro_f1_score_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    **kwargs,
):
    """
    This function is a wrapper for using block coordinate ascent
    with metric being a weighted average of instance precision and macro-averaged f1 as the target metric.
    See :func:`predict_using_bc_with_0approx` for more details and a description of parameters.
    """
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn, epsilon=1e-9):
        return (1 - alpha) * binary_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * binary_f1_score_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon) / m

    return predict_using_bc_with_0approx(
        y_proba,
        k=k,
        binary_metric_func=mixed_utility_fn,
        metric_aggregation="sum",
        skip_tn=True,
        **kwargs,
    )


def predict_optimizing_mixed_instance_precision_and_macro_balanced_accuracy_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    **kwargs,
):
    """
    This function is a wrapper for using block coordinate ascent
    with metric being a weighted average of instance precision and macro-averaged balanced_accuracy as the target metric.
    See :func:`predict_using_bc_with_0approx` for more details and a description of parameters.
    """
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn, epsilon=1e-9):
        return (1 - alpha) * binary_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * binary_balanced_accuracy_on_conf_matrix(
            tp, fp, fn, tn, epsilon=epsilon
        ) / m

    return predict_using_bc_with_0approx(
        y_proba,
        k=k,
        binary_metric_func=mixed_utility_fn,
        metric_aggregation="sum",
        skip_tn=True,
        **kwargs,
    )


def predict_optimizing_mixed_instance_precision_and_macro_jaccard_score_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    **kwargs,
):
    """
    This function is a wrapper for using block coordinate ascent
    with metric being a weighted average of instance precision and macro-averaged Jaccard score as the target metric.
    See :func:`predict_using_bc_with_0approx` for more details and a description of parameters.
    """
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn, epsilon=1e-9):
        return (1 - alpha) * binary_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * binary_jaccard_score_on_conf_matrix(
            tp, fp, fn, tn, epsilon=epsilon
        ) / m

    return predict_using_bc_with_0approx(
        y_proba,
        k=k,
        binary_metric_func=mixed_utility_fn,
        metric_aggregation="sum",
        skip_tn=True,
        **kwargs,
    )


def predict_optimizing_mixed_instance_precision_and_macro_gmean_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    **kwargs,
):
    """
    This function is a wrapper for using block coordinate ascent
    with metric being a weighted average of instance precision and macro-averaged G-mean score as the target metric.
    See :func:`predict_using_bc_with_0approx` for more details and a description of parameters.
    """
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn, epsilon=1e-9):
        return (1 - alpha) * binary_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * binary_gmean_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon) / m

    return predict_using_bc_with_0approx(
        y_proba,
        k=k,
        binary_metric_func=mixed_utility_fn,
        metric_aggregation="sum",
        skip_tn=True,
        **kwargs,
    )


def predict_optimizing_mixed_instance_precision_and_macro_hmean_using_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    alpha: float = 1,
    **kwargs,
):
    """
    This function is a wrapper for using block coordinate ascent
    with metric being a weighted average of instance precision and macro-averaged H-mean score as the target metric.
    See :func:`predict_using_bc_with_0approx` for more details and a description of parameters.
    """
    n, m = y_proba.shape

    def mixed_utility_fn(tp, fp, fn, tn, epsilon=1e-9):
        return (1 - alpha) * binary_precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * binary_hmean_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon) / m

    return predict_using_bc_with_0approx(
        y_proba,
        k=k,
        binary_metric_func=mixed_utility_fn,
        metric_aggregation="sum",
        skip_tn=True,
        **kwargs,
    )
