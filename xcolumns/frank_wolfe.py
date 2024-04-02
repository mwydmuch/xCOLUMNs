from time import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import autograd
import autograd.numpy as np
from scipy.sparse import csr_matrix

from .confusion_matrix import calculate_confusion_matrix
from .metrics import *
from .numba_csr_functions import numba_predict_weighted_per_instance_csr_step
from .types import TORCH_AVAILABLE, DefaultDataDType, DenseMatrix, DType, Matrix
from .utils import log_info, log_warning, ternary_search, uniform_search
from .weighted_prediction import predict_weighted_per_instance


if TORCH_AVAILABLE:
    import torch

    def _metric_func_with_gradient_torch(metric_func, tp, fp, fn, tn):
        tp.requires_grad_(True)
        fp.requires_grad_(True)
        fn.requires_grad_(True)
        tn.requires_grad_(True)
        value = metric_func(tp, fp, fn, tn)
        tp_grad, fp_grad, fn_grad, tn_grad = torch.autograd.grad(
            value, (tp, fp, fn, tn), materialize_grads=True, allow_unused=True
        )
        return value, tp_grad, fp_grad, fn_grad, tn_grad

    def _predict_using_randomized_weighted_classifier_torch(
        y_proba: torch.Tensor,
        k: int,
        classifiers_a: torch.Tensor,
        classifiers_b: torch.Tensor,
        classifiers_proba: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)

        n, m = y_proba.shape
        c = classifiers_proba.shape[0]
        classifiers_range = np.arange(c)
        y_pred = torch.zeros(
            y_proba.shape,
            dtype=y_proba.dtype if dtype is None else dtype,
            device=y_proba.device,
        )
        for i in range(n):
            c_i = rng.choice(classifiers_range, p=classifiers_proba)
            gains = y_proba[i] * classifiers_a[c_i] + classifiers_b[c_i]

            if k > 0:
                _, top_k = torch.topk(gains, k)
                y_pred[i, top_k] = 1
            else:
                y_pred[i, gains >= 0] = 1

        return y_pred


########################################################################################
# Randomized classifier
########################################################################################


def _predict_using_randomized_weighted_classifier_np(
    y_proba: np.ndarray,
    k: int,
    classifiers_a: np.ndarray,
    classifiers_b: np.ndarray,
    classifiers_proba: np.ndarray,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    n, m = y_proba.shape
    c = classifiers_proba.shape[0]
    classifiers_range = np.arange(c)
    y_pred = np.zeros(y_proba.shape, dtype=y_proba.dtype if dtype is None else dtype)
    for i in range(n):
        c_i = rng.choice(classifiers_range, p=classifiers_proba)
        gains = y_proba[i] * classifiers_a[c_i] + classifiers_b[c_i]

        if k > 0:
            top_k = np.argpartition(-gains, k)[:k]
            y_pred[i, top_k] = 1.0
        else:
            y_pred[i, gains > 0] = 1.0

    return y_pred


def _predict_using_randomized_weighted_classifier_csr(
    y_proba: csr_matrix,
    k: int,
    classifiers_a: np.ndarray,
    classifiers_b: np.ndarray,
    classifiers_proba: np.ndarray,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
) -> csr_matrix:
    rng = np.random.default_rng(seed)

    n, m = y_proba.shape
    c = classifiers_proba.shape[0]
    classifiers_range = np.arange(c)

    initial_row_size = k if k > 0 else 10
    y_pred_data = np.ones(
        n * initial_row_size, dtype=y_proba.data.dtype if dtype is None else dtype
    )
    y_pred_indices = np.zeros(n * initial_row_size, dtype=y_proba.indices.dtype)
    y_pred_indptr = np.arange(n + 1, dtype=y_proba.indptr.dtype) * initial_row_size

    # TODO: Can be further optimized in numba
    for i in range(n):
        c_i = rng.choice(classifiers_range, p=classifiers_proba)
        (
            y_pred_data,
            y_pred_indices,
            y_pred_indptr,
        ) = numba_predict_weighted_per_instance_csr_step(
            y_pred_data,
            y_pred_indices,
            y_pred_indptr,
            y_proba.data,
            y_proba.indices,
            y_proba.indptr,
            i,
            k,
            0.0,
            classifiers_a[c_i],
            classifiers_b[c_i],
        )

    # y_pred_data = np.ones(y_pred_indices.size, dtype=FLOAT_TYPE)

    return csr_matrix((y_pred_data, y_pred_indices, y_pred_indptr), shape=(n, m))


def predict_using_randomized_weighted_classifier(
    y_proba: Matrix,
    k: int,
    classifiers_a: DenseMatrix,
    classifiers_b: DenseMatrix,
    classifiers_proba: DenseMatrix,
    dtype: Optional[DType] = None,
    seed: Optional[int] = None,
) -> Matrix:
    r"""
    Returns the prediction of randomized weighted classifier for each instance (row) in a provided
    matrix of conditional probabilities estimates of labels :math:`\boldsymbol{H}`, (**y_proba**),
    where each element :math:`\eta_{ij} = P(y_j|x_i)`
    is the probability of the label :math:`j` for the instance :math:`i`.
    A randomized weighted classifier is a set of weighted classifiers (**classifiers_a**, and **classifiers_b**),
    one classifier is randomly selected and used for prediction for every instance according to the provided probabilities (**classifiers_proba**).

    The gains vector :math:`\boldsymbol{g}` is calculated for each instance :math:`i` as follows:

    .. math::
        c &= \text{choose random classifier index} \\
        \boldsymbol{g} &= \boldsymbol{a}_c \odot \boldsymbol{\eta}_i + \boldsymbol{b}_c

    If **k** is larger than 0, the top **k** labels with the highest gains are selected for the instance.
    If **k** is 0, then the labels with gains higher than 0 are selected for the instance.

    Args:
        y_proba: A 2D matrix of conditional probabilities for each label.
        k: The number of labels to predict for each instance.
        classifiers_a: The matrix of coeficients :math:`\boldsymbol{a}` used for calculating gains.
                       Each row represents a coeficents of a single classifier.
                       The number of rows needs to be equal to the number of rows of **classifiers_b** and size of **classifiers_proba**.
                       The number of columns needs to be equal to the number of columns of **y_proba**.
        classifiers_b: The matrix of constants :math:`\boldsymbol{a}` used for calculating gains.
                       Each row represents a constants of a single classifier.
                       The number of rows needs to be equal to the number of rows of **classifiers_b** and size of **classifiers_proba**.
                       The number of columns needs to be equal to the number of columns of **y_proba**.
        classifiers_proba: The vector of probabilities of selection for each classifier.
        dtype: The data type for the output matrix, if equal to None, the data type of **y_proba** will be used.
        seed: The seed for the random selection of classifiers.

    Returns:
        The binary prediction matrix: the shape and type of the matrix is the same as **y_proba**.
    """
    # Validate arguments

    # y_proba
    if not isinstance(y_proba, Matrix):
        raise ValueError(
            "y_proba must be either np.ndarray, torch.Tensor, or csr_matrix"
        )

    if len(y_proba.shape) == 1:
        y_proba = y_proba.reshape(1, -1)
    elif len(y_proba.shape) > 2:
        raise ValueError("y_proba must be 1d or 2d")

    # k
    if not isinstance(k, int):
        raise ValueError("k must be an integer")

    # classifiers_a, classifiers_b, classifiers_proba
    if (
        not isinstance(classifiers_a, DenseMatrix)
        or not isinstance(classifiers_b, DenseMatrix)
        or not isinstance(classifiers_proba, DenseMatrix)
    ):
        raise ValueError(
            "classifiers_a, classifiers_b, and classifiers_proba must be ndarray"
        )

    n, m = y_proba.shape
    if classifiers_a.shape[1] != m or classifiers_b.shape[1] != m:
        raise ValueError(
            "classifiers_a, classifier_b, and classifiers_proba must have the same number of columns as y_proba"
        )

    if (
        classifiers_a.shape[0] != classifiers_b.shape[0]
        or classifiers_a.shape[0] != classifiers_proba.shape[0]
    ):
        raise ValueError(
            "classifiers_a, classifier_b, and classifiers_proba must have the same number of rows"
        )

    if isinstance(y_proba, np.ndarray):
        y_pred = _predict_using_randomized_weighted_classifier_np(
            y_proba,
            k,
            classifiers_a,
            classifiers_b,
            classifiers_proba,
            dtype=dtype,
            seed=seed,
        )
    elif isinstance(y_proba, csr_matrix):
        y_pred = _predict_using_randomized_weighted_classifier_csr(
            y_proba,
            k,
            classifiers_a,
            classifiers_b,
            classifiers_proba,
            dtype=dtype,
            seed=seed,
        )
    elif TORCH_AVAILABLE and isinstance(y_proba, torch.Tensor):
        y_pred = _predict_using_randomized_weighted_classifier_torch(
            y_proba,
            k,
            classifiers_a,
            classifiers_b,
            classifiers_proba,
            dtype=dtype,
            seed=seed,
        )

    return y_pred


class RandomizedWeightedClassifier:
    """
    The class represents a randomized classifier that is a set of weighted classifiers, that are randomly selected for each instance according to the provided probabilities.
    """

    def __init__(self, k: int, a: DenseMatrix, b: DenseMatrix, p: DenseMatrix):
        r"""
        Creates a new instance of RandomizedWeightedClassifier.

        Args:
            k: The number of labels that this classifier predicts for each instance.
            a: The matrix of coeficients :math:`\boldsymbol{a}` used for calculating gains.
               Each row represents a coeficents of a single classifier.
               The number of rows needs to be equal to the number of rows of **classifiers_b** and size of **classifiers_proba**.
            b: The matrix of constants :math:`\boldsymbol{a}` used for calculating gains.
               Each row represents a constants of a single classifier.
               The number of rows needs to be equal to the number of rows of **classifiers_b** and size of **classifiers_proba**.
            p: The vector of probabilities of selection for each classifier.
        """

        # Validate arguments
        if not isinstance(k, int):
            raise ValueError("k must be an integer")

        if (
            not isinstance(a, DenseMatrix)
            or not isinstance(b, DenseMatrix)
            or not isinstance(p, DenseMatrix)
        ):
            raise ValueError("a, b, and p must be ndarray")

        if a.shape != b.shape or a.shape[0] != p.shape[0]:
            raise ValueError(
                "a, b must have the same shape and the number of rows must be equal to the number of rows of p"
            )

        self.k = k
        self.a = a
        self.b = b
        self.p = p

    def predict(
        self, y_proba: Matrix, dtype: Optional[DType] = None, seed: Optional[int] = None
    ) -> Matrix:
        r"""
        Returns the weighted prediction for each instance (row) in a provided
        matrix of conditional probabilities estimates of labels :math:`\boldsymbol{\eta}` (**y_proba**)
        using a randomized classifier.

        Args:
            y_proba: A 2D matrix of conditional probabilities for each label.
            dtype: The data type for the output matrix, if equal to None, the data type of **y_proba** will be used.
            seed: The seed for the random selection of classifiers.

        Returns:
            The binary prediction matrix: the shape and type of the matrix is the same as **y_proba**.
        """

        # Additional arguments checks with nicer error messages
        if y_proba.shape[1] != self.a.shape[1] or y_proba.shape[1] != self.b.shape[1]:
            raise ValueError(
                f"This classifier support the input matrix with {self.a.shape[1]} columns (labels), got {y_proba.shape[1]}"
            )

        return predict_using_randomized_weighted_classifier(
            y_proba, self.k, self.a, self.b, self.p, dtype=dtype, seed=seed
        )


########################################################################################
# Frank-Wolfe algorithm
########################################################################################


def _metric_func_with_gradient_autograd(metric_func, tp, fp, fn, tn):
    grad_func = autograd.grad(metric_func, argnum=[0, 1, 2, 3])
    return float(metric_func(tp, fp, fn, tn)), *grad_func(tp, fp, fn, tn)


def _find_best_alpha(
    metric_func: Callable,
    tp: DenseMatrix,
    fp: DenseMatrix,
    fn: DenseMatrix,
    tn: DenseMatrix,
    tp_i: DenseMatrix,
    fp_i: DenseMatrix,
    fn_i: DenseMatrix,
    tn_i: DenseMatrix,
    search_algo: str = "uniform",
    eps: float = 0.001,
    uniform_search_step: float = 0.001,
) -> Tuple[float, float]:
    conf_mat_comb = lambda alpha: metric_func(
        (1 - alpha) * tp + alpha * tp_i,
        (1 - alpha) * fp + alpha * fp_i,
        (1 - alpha) * fn + alpha * fn_i,
        (1 - alpha) * tn + alpha * tn_i,
    )
    if search_algo == "uniform":
        return uniform_search(0, 1, uniform_search_step, conf_mat_comb)
    elif search_algo == "ternary":
        return ternary_search(0, 1, eps, conf_mat_comb)
    else:
        raise ValueError(f"Unknown search algorithm {search_algo}")


def find_classifier_using_fw(
    y_true: Matrix,
    y_proba: Matrix,
    metric_func: Callable,
    k: int,
    max_iters: int = 100,
    init_classifier: Union[str, Tuple[DenseMatrix, DenseMatrix]] = "random",  # or "top"
    maximize=True,
    search_for_best_alpha: bool = True,
    alpha_search_algo: str = "uniform",  # or "ternary"
    alpha_eps: float = 0.001,
    alpha_uniform_search_step: float = 0.001,
    skip_tn=False,
    seed=None,
    verbose: bool = False,
    return_meta: bool = False,
) -> Union[
    RandomizedWeightedClassifier, Tuple[RandomizedWeightedClassifier, Dict[str, Any]]
]:
    r"""
    Finds a randomized classifier that optimizes the given metric using the Frank-Wolfe algorithm
    on provided training dataset of true labels **y_true** and corresponding conditional probabilities **y_proba**.

    The algorithm iteratively calculates the gradient of the metric with respect to the confusion matrix and updates the randomized classfuer accordingly.

    See for more details:
    > [Erik Schultheis, Wojciech Kotłowski, Marek Wydmuch, Rohit Babbar, Strom Borman, Krzysztof Dembczyński. Consistent algorithms for multi-label classification with macro-at-k metrics. ICLR 2024.](https://arxiv.org/abs/2401.16594)

    Args:
        y_true: A 2D matrix of true labels of set that will be used to find the optimal classifier.
        y_proba: A 2D matrix of conditional probabilities that will be used to find the optimal classifier.
        metric_func: The metric function defined on confusion matrix to optimize.
        k: The budget of labels to predict for each instance.
           If equal to 0, this means that there is no budget constraint.
        max_iters: The maximum number of iterations.
        init_classifier: The initial classifier, can be either "random", "top", or an initial weighted classifier with provided vectors of coeficients :math:`\boldsymbol{a}` and constants :math:`\boldsymbol{b}`.
        maximize: Whether to maximize or minimize the metric.
        search_for_best_alpha: Whether to search for the best alpha (step size) in each iteration or to use standard Frank-Wolfe step size :math:`2/(i + 1)`, where :math:`i` is an iteration number.
                               Setting slows down the algorithm, but may help to find better solution if the metric is not convex.
        alpha_search_algo: The algorithm for searching for the best alpha, can be either "uniform" or "ternary".
                           "Ternary" should be only used if the metric is unimodal.
        alpha_eps: The stopping condition, if the alpha is smaller than this value, the algorithm stops.
        alpha_uniform_search_step: The step size for uniform search of alpha.
        skip_tn: Whether to skip the calculation of True Negatives in the confusion matrix, if the metric does not use the True Negatives, this can speed up the calculation, especially when using sparse matrices.
        seed: The seed for the random selection of classifiers.
        verbose: Whether to print additional information.
        return_meta: Whether to return meta data.

    Returns:
        The :class:`RandomizedWeightedClassifier`. If **return_meta** is True, additionally, a dictionary is returned, that contains the time taken to calculate the prediction, the number of iterations, and step sizes for each iteration and calculated metric values for each weighted classifier.
    """

    log_info(
        "Starting searching for optimal randomized classifier using Frank-Wolfe algorithm ...",
        verbose,
    )

    alpha_eps = alpha_uniform_search_step

    # Validate y_true and y_proba
    if type(y_true) != type(y_proba) and isinstance(y_true, Matrix):
        raise ValueError(
            f"y_true and y_proba have unsupported combination of types {type(y_true)} and {type(y_proba)}, should be both np.ndarray, both torch.Tensor, or both csr_matrix"
        )

    if y_true.shape != y_proba.shape:
        raise ValueError(
            f"y_true and y_proba must have the same shape, got {y_true.shape} and {y_proba.shape}"
        )

    n, m = y_proba.shape

    log_info(
        f"  Initializing initial {init_classifier if isinstance(init_classifier, str) else 'custom'} classifier ...",
        verbose,
    )
    # Initialize the classifiers matrix
    rng = np.random.default_rng(seed)
    classifiers_a = np.zeros((max_iters + 1, m), dtype=DefaultDataDType)
    classifiers_b = np.zeros((max_iters + 1, m), dtype=DefaultDataDType)
    classifiers_proba = np.ones(max_iters + 1, dtype=DefaultDataDType)

    if init_classifier == "top":
        classifiers_a[0] = np.ones(m, dtype=DefaultDataDType)
        classifiers_b[0] = np.full(m, -0.5, dtype=DefaultDataDType)
    elif init_classifier == "random":
        classifiers_a[0] = rng.random(m)
        classifiers_b[0] = rng.random(m) - 0.5
    elif (
        isinstance(init_classifier, (tuple, list))
        and len(init_classifier) == 2
        and isinstance(init_classifier[0], DenseMatrix)
        and isinstance(init_classifier[1], DenseMatrix)
        and init_classifier[0].shape == (m,)
        and init_classifier[1].shape == (m,)
    ):
        # TODO: This from torch -> numpy and back to torch is not nice, maybe improve it later
        if TORCH_AVAILABLE and isinstance(init_classifier[0], torch.Tensor):
            classifiers_a[0] = init_classifier[0].cpu().numpy()
        else:
            classifiers_a[0] = init_classifier[0]

        if TORCH_AVAILABLE and isinstance(init_classifier[1], torch.Tensor):
            classifiers_b[0] = init_classifier[1].cpu().numpy()
        else:
            classifiers_b[0] = init_classifier[1]
    else:
        raise ValueError(
            f"Unsupported type of init_classifier, it should be in ['random', 'top'], or a tuple of two np.ndarray or torch.Tensor of shape (y_true.shape[1], )"
        )

    # Adjust types according to the type of y_true and y_proba
    if isinstance(y_true, (np.ndarray, csr_matrix)):
        metric_func_with_gradient = _metric_func_with_gradient_autograd
    elif TORCH_AVAILABLE and isinstance(y_true, torch.Tensor):
        metric_func_with_gradient = _metric_func_with_gradient_torch
        classifiers_a = torch.tensor(
            classifiers_a, dtype=y_proba.dtype, device=y_proba.device
        )
        classifiers_b = torch.tensor(
            classifiers_b, dtype=y_proba.dtype, device=y_proba.device
        )
        classifiers_proba = torch.tensor(
            classifiers_proba, dtype=y_proba.dtype, device=y_proba.device
        )

    y_pred_i = predict_weighted_per_instance(
        y_proba, k, th=0.0, a=classifiers_a[0], b=classifiers_b[0]
    )

    tp, fp, fn, tn = calculate_confusion_matrix(
        y_true, y_pred_i, normalize=True, skip_tn=skip_tn
    )
    utility_i = metric_func(tp, fp, fn, tn)

    if return_meta:
        meta = {
            "alphas": [],
            "classifiers_utilities": [],
            "utilities": [],
            "time": time(),
        }
        meta["utilities"].append(utility_i)
        meta["classifiers_utilities"].append(utility_i)

    for i in range(1, max_iters + 1):
        log_info(f"  Starting iteration {i}/{max_iters} ...", verbose)
        old_utility, Gtp, Gfp, Gfn, Gtn = metric_func_with_gradient(
            metric_func, tp, fp, fn, tn
        )

        classifiers_a[i] = Gtp - Gfp - Gfn + Gtn
        classifiers_b[i] = Gfp - Gtn
        if not maximize:
            classifiers_a[i] *= -1
            classifiers_b[i] *= -1

        y_pred_i = predict_weighted_per_instance(
            y_proba, k, th=0.0, a=classifiers_a[i], b=classifiers_b[i]
        )
        tp_i, fp_i, fn_i, tn_i = calculate_confusion_matrix(
            y_true, y_pred_i, normalize=True, skip_tn=skip_tn
        )
        utility_i = metric_func(tp_i, fp_i, fn_i, tn_i)

        if search_for_best_alpha:
            alpha, _ = _find_best_alpha(
                metric_func,
                tp,
                fp,
                fn,
                tn,
                tp_i,
                fp_i,
                fn_i,
                tn_i,
                search_algo=alpha_search_algo,
                eps=alpha_eps,
                uniform_search_step=alpha_uniform_search_step,
            )
        else:
            alpha = 2 / (i + 1)

        tp = (1 - alpha) * tp + alpha * tp_i
        fp = (1 - alpha) * fp + alpha * fp_i
        fn = (1 - alpha) * fn + alpha * fn_i
        tn = (1 - alpha) * tn + alpha * tn_i
        new_utility = metric_func(tp, fp, fn, tn)

        log_info(
            f"    Iteration {i}/{max_iters} finished, alpha: {alpha}, utility: {old_utility} -> {new_utility}",
            verbose,
        )

        if alpha < alpha_eps:
            log_info(f"  Stopping because alpha is smaller than {alpha_eps}", verbose)
            # Truncate unused classifiers
            classifiers_a = classifiers_a[:i]
            classifiers_b = classifiers_b[:i]
            classifiers_proba = classifiers_proba[:i]
            break

        if return_meta:
            meta["alphas"].append(alpha)
            meta["classifiers_utilities"].append(utility_i)
            meta["utilities"].append(new_utility)
            meta["iters"] = i

        classifiers_proba[:i] *= 1 - alpha
        classifiers_proba[i] = alpha

    rnd_classifier = RandomizedWeightedClassifier(
        k, classifiers_a, classifiers_b, classifiers_proba
    )

    if return_meta:
        meta["time"] = time() - meta["time"]
        return (
            rnd_classifier,
            meta,
        )
    else:
        return rnd_classifier


########################################################################################
# Wrapper functions of Frank Wolfe algorithm for specific metrics
########################################################################################


def _make_frank_wolfe_wrapper(
    metric_func: Callable,
    metric_name: str,
    maximize: bool = True,
    skip_tn: bool = False,
    warn_k_eq_0: bool = False,
):
    def find_classifier_for_metric_using_fw(
        y_true: Matrix, y_proba: Matrix, k: int, **kwargs
    ):
        if warn_k_eq_0 and k == 0:
            log_warning(
                f"Warning: k=0 results in degenerated solution for {metric_name}!",
            )

        return find_classifier_using_fw(
            y_true,
            y_proba,
            metric_func,
            k,
            maximize=maximize,
            skip_tn=skip_tn,
            **kwargs,
        )

    find_classifier_for_metric_using_fw.__doc__ = f"""
    Find a randomized classifier that maximizes {metric_name} metric using Frank-Wolfe algorithm.
    It is equivalent to calling ``find_classifier_using_fw(y_true, y_proba, {metric_func.__name__}, k, ..., maximize={maximize}, skip_tn={skip_tn})`` function.
    See :meth:`find_classifier_using_fw` for more details and a description of arguments.
    """

    return add_kwargs_to_signature(
        find_classifier_for_metric_using_fw,
        find_classifier_using_fw,
        skip=["metric_func", "maximize", "skip_tn"],
    )


find_classifier_optimizing_macro_precision_using_fw = _make_frank_wolfe_wrapper(
    macro_precision,
    "macro-averaged precision",
    maximize=True,
    skip_tn=True,
    warn_k_eq_0=True,
)
find_classifier_optimizing_macro_recall_using_fw = _make_frank_wolfe_wrapper(
    macro_recall, "macro-averaged recall", maximize=True, skip_tn=True, warn_k_eq_0=True
)

find_classifier_optimizing_macro_f1_using_fw = _make_frank_wolfe_wrapper(
    macro_f1_score, "macro-averaged F1 score", maximize=True, skip_tn=True
)
find_classifier_optimizing_micro_f1_using_fw = _make_frank_wolfe_wrapper(
    micro_f1_score, "micro-averaged F1 score", maximize=True, skip_tn=True
)

find_classifier_optimizing_macro_jaccard_score_using_fw = _make_frank_wolfe_wrapper(
    macro_balanced_accuracy, "macro-averaged Jaccard score", maximize=True, skip_tn=True
)
find_classifier_optimizing_micro_jaccard_score_using_fw = _make_frank_wolfe_wrapper(
    micro_jaccard_score, "micro-averaged Jaccard score", maximize=True, skip_tn=True
)

find_classifier_optimizing_macro_balanced_accuracy_using_fw = _make_frank_wolfe_wrapper(
    macro_balanced_accuracy, "macro-averaged balanced accuracy", maximize=True
)
find_classifier_optimizing_micro_balanced_accuracy_using_fw = _make_frank_wolfe_wrapper(
    micro_balanced_accuracy, "micro-averaged balanced accuracy", maximize=True
)

find_classifier_optimizing_macro_hmean_using_fw = _make_frank_wolfe_wrapper(
    macro_hmean, "macro-averaged H-mean", maximize=True
)
find_classifier_optimizing_micro_hmean_using_fw = _make_frank_wolfe_wrapper(
    macro_hmean, "micro-averaged H-mean", maximize=True
)

find_classifier_optimizing_macro_gmean_using_fw = _make_frank_wolfe_wrapper(
    macro_gmean, "macro-averaged G-mean", maximize=True
)
find_classifier_optimizing_micro_gmean_using_fw = _make_frank_wolfe_wrapper(
    macro_gmean, "micro-averaged G-mean", maximize=True
)
