from time import time

import numpy as np
from custom_utilities_methods import *
from scipy.sparse import csc_matrix, csr_matrix

from xcolumns.block_coordinate import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics import *
from xcolumns.types import *
from xcolumns.utils import *
from xcolumns.weighted_prediction import *


def binary_min_tp_tn(tp, fp, fn, tn):
    return np.fmin(tp, tn)


def find_binary_threshold(
    sorted_true: np.ndarray, sorted_proba: np.ndarray, binary_utility_func
):
    n = sorted_true.shape[0]

    pred = np.zeros(n, dtype=FLOAT_TYPE)
    best_utility = -np.inf
    best_threshold = 0

    tp = 0
    fp = 0
    fn = sorted_true.sum()
    tn = n - fn
    for i in range(n):
        pred[i] = 1.0
        tp += sorted_true[i]
        fp += 1 - sorted_true[i]
        fn -= sorted_true[i]
        tn -= 1 - sorted_true[i]
        if pred[i] == pred[i + 1]:
            continue

        utility = binary_utility_func(tp, fp, fn, tn)
        if utility > best_utility:
            best_utility = utility
            if i == n - 1:
                best_threshold = sorted_proba[i] - 1e-6
            else:
                best_threshold = (sorted_proba[i] + sorted_proba[i + 1]) / 2

    return best_threshold, best_utility


def find_thresholds_np(y_true, y_proba, binary_utility_func, **kwargs):
    n, m = y_true.shape
    thresholds = np.zeros(m, dtype=FLOAT_TYPE)
    utilities = np.zeros(m, dtype=FLOAT_TYPE)
    for j in trange(m):
        sorted_order = np.argsort(-y_proba[:, j])
        sorted_proba = y_proba[sorted_order, j]
        sorted_true = y_true[sorted_order, j]
        thresholds[j], utilities[j] = find_binary_threshold(
            sorted_true, sorted_proba, binary_utility_func
        )

    return thresholds, utilities


def find_thresholds_sparse(y_true, y_proba, binary_utility_func, **kwargs):
    if not isinstance(y_true, csc_matrix):
        y_true = csc_matrix(y_true)

    if not isinstance(y_proba, csc_matrix):
        y_proba = csc_matrix(y_proba)

    n, m = y_true.shape
    thresholds = np.zeros(m, dtype=FLOAT_TYPE)
    utilities = np.zeros(m, dtype=FLOAT_TYPE)
    for j in trange(m):
        proba = np.zeros(n, dtype=FLOAT_TYPE)
        true = np.zeros(n, dtype=FLOAT_TYPE)

        thresholds[j], utilities[j] = find_binary_threshold(
            sorted_true, sorted_proba, binary_utility_func
        )

    return thresholds, utilities


def find_thresholds(y_true, y_proba, binary_utility_func, return_meta=True, **kwargs):
    assert y_true.shape == y_proba.shape

    print(f"Finding thresholds with {y_true.shape} and {y_proba.shape} ...")
    if return_meta:
        meta = {"time": time(), "iters": 1}

    if isinstance(y_true, csr_matrix) and isinstance(y_proba, csr_matrix):
        thresholds, utilites = find_thresholds_sparse(
            y_true, y_proba, binary_utility_func, **kwargs
        )
    elif isinstance(y_true, np.ndarray) and isinstance(y_proba, np.ndarray):
        thresholds, utilites = find_thresholds_np(
            y_true, y_proba, binary_utility_func, **kwargs
        )
    else:
        raise ValueError("y_true and y_proba must be both dense or both sparse")

    if return_meta:
        meta["time"] = time() - meta["time"]
        meta["thresholds"] = thresholds
        meta["utilites"] = utilites
        return thresholds, meta
    else:
        return thresholds


def find_thresholds_wrapper(
    y_proba, binary_utility_func, k, y_true_valid=None, y_proba_valid=None, **kwargs
):
    if y_true_valid is None:
        return no_support(y_proba)

    thresholds, meta = find_thresholds(
        y_true_valid, y_proba_valid, binary_utility_func, **kwargs
    )
    if k > 0:
        y_pred = np.zeros(y_proba.shape, dtype=FLOAT_TYPE)
        for i in range(y_proba.shape[0]):
            gains = y_proba[i] - thresholds
            top_k = np.argpartition(-gains, k)[:k]
            y_pred[i, top_k] = 1.0
    else:
        y_pred = y_proba >= thresholds

    return y_pred, meta


def default_prediction(y_proba, k, **kwargs):
    y_pred, meta = predict_weighted_per_instance(
        y_proba,
        k,
        th=0.5,
        a=None,
        b=None,
        return_meta=True,
    )
    return y_pred, meta


def find_thresholds_macro_f1(
    y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return find_thresholds_wrapper(
        y_proba,
        binary_f1_score_on_conf_matrix,
        k,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        **kwargs,
    )


def find_thresholds_macro_f1_on_test(
    y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return find_thresholds_wrapper(
        y_proba,
        binary_f1_score_on_conf_matrix,
        k,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        **kwargs,
    )


def find_thresholds_macro_f1_etu(
    y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return find_thresholds_wrapper(
        y_proba,
        binary_f1_score_on_conf_matrix,
        k,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        **kwargs,
    )


def find_thresholds_macro_min_tp_tn(
    y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return find_thresholds_wrapper(
        y_proba,
        binary_min_tp_tn,
        k,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        **kwargs,
    )


def find_thresholds_macro_min_tp_tn_on_test(
    y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return find_thresholds_wrapper(
        y_proba,
        binary_min_tp_tn,
        k,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        **kwargs,
    )


def find_thresholds_macro_min_tp_tn_etu(
    y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return find_thresholds_wrapper(
        y_proba,
        binary_min_tp_tn,
        k,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        **kwargs,
    )
