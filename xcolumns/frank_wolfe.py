from time import time
from typing import Callable, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix

from .default_types import *
from .metrics_on_conf_matrix import *
from .utils import *
from .weighted_prediction import predict_weighted_per_instance


def _get_grad_as_numpy(t):
    if t.grad is not None:
        return t.grad.numpy()
    else:
        return np.zeros(t.shape, dtype=FLOAT_TYPE)


def utility_func_with_gradient(utility_func, tp, fp, fn, tn):
    if not isinstance(tp, torch.Tensor) or not tp.requires_grad:
        tp = torch.tensor(tp, requires_grad=True, dtype=TORCH_FLOAT_TYPE)
        fp = torch.tensor(fp, requires_grad=True, dtype=TORCH_FLOAT_TYPE)
        fn = torch.tensor(fn, requires_grad=True, dtype=TORCH_FLOAT_TYPE)
        tn = torch.tensor(tn, requires_grad=True, dtype=TORCH_FLOAT_TYPE)
    utility = utility_func(tp, fp, fn, tn)
    utility.backward()
    return (
        float(utility),
        _get_grad_as_numpy(tp),
        _get_grad_as_numpy(fp),
        _get_grad_as_numpy(fn),
        _get_grad_as_numpy(tn),
    )


def _find_best_alpha(
    utility_func,
    tp,
    fp,
    fn,
    tn,
    tp_i,
    fp_i,
    fn_i,
    tn_i,
    search_algo="lin",
    eps=0.001,
    lin_search_step=0.001,
):
    conf_mat_comb = lambda alpha: utility_func(
        (1 - alpha) * tp + alpha * tp_i,
        (1 - alpha) * fp + alpha * fp_i,
        (1 - alpha) * fn + alpha * fn_i,
        (1 - alpha) * tn + alpha * tn_i,
    )
    if search_algo == "lin":
        return lin_search(0, 1, lin_search_step, conf_mat_comb)
    elif search_algo == "bin":
        return bin_search(0, 1, eps, conf_mat_comb)
    elif search_algo == "ternary":
        return ternary_search(0, 1, eps, conf_mat_comb)
    else:
        raise ValueError(f"Unknown search algorithm {search_algo}")


def find_optimal_randomized_classifier_using_frank_wolfe(
    y_true: Union[np.ndarray, csr_matrix],
    y_proba: Union[np.ndarray, csr_matrix],
    utility_func: Callable,
    k: int,
    max_iters: int = 100,
    init_classifier: Union[
        str, Tuple[np.ndarray, np.ndarray]
    ] = "random",  # or "random"
    search_for_best_alpha: bool = True,
    alpha_search_algo: str = "lin",
    alpha_eps: float = 0.001,
    alpha_lin_search_step: float = 0.001,
    skip_tn=False,
    seed=None,
    verbose: bool = True,
    return_meta: bool = False,
    **kwargs,
):
    log = print
    if not verbose:
        log = lambda *args, **kwargs: None

    if type(y_true) != type(y_proba):
        raise ValueError(
            f"y_true and y_proba have unsupported combination of types {type(y_true)}, {type(y_proba)}, should be both np.ndarray or both csr_matrix"
        )

    log("Starting Frank-Wolfe algorithm")
    n, m = y_proba.shape

    rng = np.random.default_rng(seed)
    classifiers_a = np.zeros((max_iters, m), dtype=FLOAT_TYPE)
    classifiers_b = np.zeros((max_iters, m), dtype=FLOAT_TYPE)
    classifiers_proba = np.ones(max_iters, dtype=FLOAT_TYPE)

    log(f"  Initializing initial {init_classifier} classifier ...")
    if init_classifier in ["default", "topk"]:
        classifiers_a[0] = np.ones(m, dtype=FLOAT_TYPE)
        classifiers_b[0] = np.full(m, -0.5, dtype=FLOAT_TYPE)
    elif init_classifier == "random":
        classifiers_a[0] = rng.random(m)
        classifiers_b[0] = rng.random(m) - 0.5
    else:
        raise ValueError(f"Unsupported type of init_classifier: {init_classifier}")
    y_pred_i = predict_weighted_per_instance(
        y_proba, k, th=0.0, a=classifiers_a[0], b=classifiers_b[0]
    )

    log(
        f"    y_true: {y_true.shape}, y_pred: {y_pred_i.shape}, y_proba: {y_proba.shape}"
    )

    tp, fp, fn, tn = calculate_confusion_matrix(
        y_true, y_pred_i, normalize=True, skip_tn=skip_tn
    )
    utility_i = float(utility_func(tp, fp, fn, tn))

    if return_meta:
        meta = {
            "alphas": [],
            "classifiers_utilities": [],
            "utilities": [],
            "time": time(),
        }
        meta["utilities"].append(utility_i)
        meta["classifiers_utilities"].append(utility_i)

    for i in range(1, max_iters):
        log(f"  Starting iteration {i} ...")
        old_utility, Gtp, Gfp, Gfn, Gtn = utility_func_with_gradient(
            utility_func, tp, fp, fn, tn
        )

        classifiers_a[i] = Gtp - Gfp - Gfn + Gtn
        classifiers_b[i] = Gfp - Gtn
        y_pred_i = predict_weighted_per_instance(
            y_proba, k, th=0.0, a=classifiers_a[i], b=classifiers_b[i]
        )
        tp_i, fp_i, fn_i, tn_i = calculate_confusion_matrix(
            y_true, y_pred_i, normalize=True, skip_tn=skip_tn
        )
        utility_i = float(utility_func(tp_i, fp_i, fn_i, tn_i))

        if search_for_best_alpha:
            alpha, _ = _find_best_alpha(
                utility_func,
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
                lin_search_step=alpha_lin_search_step,
            )
        else:
            alpha = 2 / (i + 1)

        classifiers_proba[:i] *= 1 - alpha
        classifiers_proba[i] = alpha
        tp = (1 - alpha) * tp + alpha * tp_i
        fp = (1 - alpha) * fp + alpha * fp_i
        fn = (1 - alpha) * fn + alpha * fn_i
        tn = (1 - alpha) * tn + alpha * tn_i

        new_utility = float(utility_func(tp, fp, fn, tn))

        if return_meta:
            meta["alphas"].append(alpha)
            meta["classifiers_utilities"].append(utility_i)
            meta["utilities"].append(new_utility)
            meta["iters"] = i

        log(
            f"    Iteration {i} finished, alpha: {alpha}, utility: {old_utility} -> {new_utility}"
        )

        if alpha < alpha_eps:
            print(f"    Stopping because alpha is smaller than {alpha_eps}")
            # Truncate unused classifiers
            classifiers_a = classifiers_a[:i]
            classifiers_b = classifiers_b[:i]
            classifiers_proba = classifiers_proba[:i]
            break

    if return_meta:
        meta["time"] = time() - meta["time"]
        return (
            RandomizedWeightedClassifier(
                classifiers_a, classifiers_b, classifiers_proba
            ),
            meta,
        )
    else:
        return RandomizedWeightedClassifier(
            classifiers_a, classifiers_b, classifiers_proba
        )


def _predict_using_randomized_classifier_np(
    y_proba, classifiers_a, classifiers_b, classifiers_proba, k, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    n, m = y_proba.shape
    c = classifiers_proba.shape[0]
    classifiers_range = np.arange(c)
    result = np.zeros(y_proba.shape, dtype=FLOAT_TYPE)
    for i in range(n):
        c_i = np.random.choice(classifiers_range, p=classifiers_proba)
        gains = y_proba[i] * classifiers_a[c_i] + classifiers_b[c_i]

        if k > 0:
            top_k = np.argpartition(-gains, k)[:k]
            result[i, top_k] = 1.0
        else:
            result[i, gains > 0] = 1.0

    return result


def _predict_using_randomized_classifier_csr(
    y_proba, classifiers_a, classifiers_b, classifiers_proba, k, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    n, m = y_proba.shape
    c = classifiers_proba.shape[0]
    y_pred_indices = np.zeros(n * max(1, k), dtype=IND_TYPE)
    y_pred_indptr = np.zeros(n + 1, dtype=IND_TYPE)
    classifiers_range = np.arange(c)

    # TODO: Can be futher optimized in numba
    for i in range(n):
        c_i = np.random.choice(classifiers_range, p=classifiers_proba)
        y_pred_indices, y_pred_indptr = numba_predict_weighted_per_instance_csr_step(
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

    y_pred_data = np.ones(y_pred_indices.size, dtype=FLOAT_TYPE)

    return csr_matrix((y_pred_data, y_pred_indices, y_pred_indptr), shape=(n, m))


def predict_using_randomized_classifier(
    y_proba, classifiers_a, classifiers_b, classifiers_proba, k, seed=None
):
    if not isinstance(y_proba, (np.ndarray, csr_matrix)):
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    if (
        not isinstance(classifiers_a, np.ndarray)
        or not isinstance(classifiers_b, np.ndarray)
        or not isinstance(classifiers_proba, np.ndarray)
    ):
        raise ValueError(
            "classifiers_a, classifiers_b, and classifiers_proba must be ndarray"
        )

    if len(y_proba.shape) == 1:
        y_proba = y_proba.reshape(1, -1)
    elif len(y_proba.shape) > 2:
        raise ValueError("y_proba must be 1d or 2d")

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
        return _predict_using_randomized_classifier_np(
            y_proba, classifiers_a, classifiers_b, classifiers_proba, k, seed=seed
        )
    elif isinstance(y_proba, csr_matrix):
        return _predict_using_randomized_classifier_csr(
            y_proba, classifiers_a, classifiers_b, classifiers_proba, k, seed=seed
        )


class RandomizedWeightedClassifier:
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p

    def predict(self, y_proba, k, seed=None) -> Union[np.ndarray, torch.tensor, csr_matrix]:
        return predict_using_randomized_classifier(
            y_proba, self.a, self.b, self.p, k, seed=seed
        )
