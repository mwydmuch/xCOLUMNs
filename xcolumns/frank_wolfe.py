import logging
from time import time
from typing import Callable, Tuple, Union

import autograd
import autograd.numpy as np
from scipy.sparse import csr_matrix

from .confusion_matrix import calculate_confusion_matrix
from .metrics import *
from .types import *
from .utils import *
from .weighted_prediction import predict_weighted_per_instance


_torch_available = False
try:
    import torch

    def metric_func_with_gradient_torch(metric_func, tp, fp, fn, tn):
        tp.requires_grad_(True)
        fp.requires_grad_(True)
        fn.requires_grad_(True)
        tn.requires_grad_(True)
        value = metric_func(tp, fp, fn, tn)
        tp_grad, fp_grad, fn_grad, tn_grad = torch.autograd.grad(
            value, (tp, fp, fn, tn), materialize_grads=True, allow_unused=True
        )
        return value, tp_grad, fp_grad, fn_grad, tn_grad

    def _predict_using_randomized_classifier_torch(
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

    _torch_available = True
except ImportError:
    pass


def metric_func_with_gradient_autograd(metric_func, tp, fp, fn, tn):
    grad_func = autograd.grad(metric_func, argnum=[0, 1, 2, 3])
    return float(metric_func(tp, fp, fn, tn)), *grad_func(tp, fp, fn, tn)


def _find_best_alpha(
    metric_func,
    tp,
    fp,
    fn,
    tn,
    tp_i,
    fp_i,
    fn_i,
    tn_i,
    search_algo="uniform",
    eps=0.001,
    uniform_search_step=0.001,
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


def find_optimal_randomized_classifier_using_frank_wolfe(
    y_true: Matrix,
    y_proba: Matrix,
    metric_func: Callable,
    k: int,
    max_iters: int = 100,
    init_classifier: Union[
        str, Tuple[DenseMatrix, DenseMatrix]
    ] = "random",  # or "default", "topk"
    maximize=True,
    search_for_best_alpha: bool = True,
    alpha_search_algo: str = "uniform",  # or "ternary"
    alpha_eps: float = 0.001,
    alpha_uniform_search_step: float = 0.001,
    skip_tn=False,
    seed=None,
    verbose: bool = False,
    return_meta: bool = False,
):
    log_info(
        "Starting searching for optimal randomized classifier using Frank-Wolfe algorithm ...",
        verbose,
    )

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

    log_info(f"  Initializing initial {init_classifier} classifier ...", verbose)
    # Initialize the classifiers matrix
    rng = np.random.default_rng(seed)
    classifiers_a = np.zeros((max_iters, m), dtype=DefaultDataDType)
    classifiers_b = np.zeros((max_iters, m), dtype=DefaultDataDType)
    classifiers_proba = np.ones(max_iters, dtype=DefaultDataDType)

    if init_classifier in ["default", "topk"]:
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
        if _torch_available and isinstance(init_classifier[0], torch.Tensor):
            classifiers_a[0] = init_classifier[0].numpy()
        else:
            classifiers_a[0] = init_classifier[0]

        if _torch_available and isinstance(init_classifier[1], torch.Tensor):
            classifiers_b[0] = init_classifier[1].numpy()
        else:
            classifiers_b[0] = init_classifier[1]
    else:
        raise ValueError(
            f"Unsupported type of init_classifier, it should be 'default', 'topk', 'random', or a tuple of two np.ndarray or torch.Tensor of shape (y_true.shape[1], )"
        )

    # Adjust types according to the type of y_true and y_proba
    if isinstance(y_true, (np.ndarray, csr_matrix)):
        metric_func_with_gradient = metric_func_with_gradient_autograd
    elif _torch_available and isinstance(y_true, torch.Tensor):
        metric_func_with_gradient = metric_func_with_gradient_torch
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

        classifiers_proba[:i] *= 1 - alpha
        classifiers_proba[i] = alpha
        tp = (1 - alpha) * tp + alpha * tp_i
        fp = (1 - alpha) * fp + alpha * fp_i
        fn = (1 - alpha) * fn + alpha * fn_i
        tn = (1 - alpha) * tn + alpha * tn_i

        new_utility = metric_func(tp, fp, fn, tn)

        if return_meta:
            meta["alphas"].append(alpha)
            meta["classifiers_utilities"].append(utility_i)
            meta["utilities"].append(new_utility)
            meta["iters"] = i

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


def _predict_using_randomized_classifier_np(
    y_proba: np.ndarray,
    k: int,
    classifiers_a: np.ndarray,
    classifiers_b: np.ndarray,
    classifiers_proba: np.ndarray,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
):
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


def _predict_using_randomized_classifier_csr(
    y_proba: csr_matrix,
    k: int,
    classifiers_a: np.ndarray,
    classifiers_b: np.ndarray,
    classifiers_proba: np.ndarray,
    dtype: Optional[np.dtype] = None,
    seed: Optional[int] = None,
):
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


def predict_using_randomized_classifier(
    y_proba: Matrix,
    k: int,
    classifiers_a: DenseMatrix,
    classifiers_b: DenseMatrix,
    classifiers_proba: DenseMatrix,
    dtype: Optional[DType] = None,
    seed: Optional[int] = None,
) -> Matrix:
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
        y_pred = _predict_using_randomized_classifier_np(
            y_proba,
            k,
            classifiers_a,
            classifiers_b,
            classifiers_proba,
            dtype=dtype,
            seed=seed,
        )
    elif isinstance(y_proba, csr_matrix):
        y_pred = _predict_using_randomized_classifier_csr(
            y_proba,
            k,
            classifiers_a,
            classifiers_b,
            classifiers_proba,
            dtype=dtype,
            seed=seed,
        )
    elif _torch_available and isinstance(y_proba, torch.Tensor):
        y_pred = _predict_using_randomized_classifier_torch(
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
    def __init__(self, k, a, b, p):
        self.k = k
        self.a = a
        self.b = b
        self.p = p

    def predict(self, y_proba: Matrix, seed: Optional[int] = None) -> Matrix:
        return predict_using_randomized_classifier(
            y_proba, self.k, self.a, self.b, self.p, seed=seed
        )
