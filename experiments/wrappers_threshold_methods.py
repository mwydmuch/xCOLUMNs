import numpy as np
from time import time

from xcolumns.block_coordinate import *
from xcolumns.metrics import *
from xcolumns.weighted_prediction import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics_on_conf_matrix import *
from xcolumns.default_types import *
from xcolumns.utils import *


from custom_utilities_methods import *


def bin_min_tp_tn(tp, fp, fn, tn):
    return np.fmin(tp, tn)


def find_thresholds(y_true, y_proba, bin_utility_func, **kwargs):
    assert y_true.shape == y_proba.shape

    n, m = y_proba.shape
    thresholds = np.zeros(m, dtype=FLOAT_TYPE)

    meta = {"time": time(), "iters": 1}
    for j in trange(m):
        sorted_order = np.argsort(-y_proba[:, j])
        sorted_proba = y_proba[sorted_order, j]
        sorted_true = y_true[sorted_order, j]
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
            utility = bin_utility_func(tp, fp, fn, tn)
            if utility > best_utility:
                best_utility = utility
                best_threshold = sorted_proba[i] - 1e-6

        thresholds[j] = best_threshold

    meta["time"] = time() - meta["time"]
    meta["thresholds"] = thresholds

    return thresholds, meta


def find_thresholds_wrapper(
    y_proba, bin_utility_func, k, y_true_valid=None, y_proba_valid=None, **kwargs
):
    thresholds, meta = find_thresholds(y_true_valid, y_proba_valid, bin_utility_func, **kwargs)
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
    meta = {"time": time(), "iters": 1}
    meta["time"] = time() - meta["time"]
    if k > 0:
        y_pred = np.zeros(y_proba.shape, dtype=FLOAT_TYPE)
        for i in range(y_proba.shape[0]):
            top_k = np.argpartition(-y_proba[i], k)[:k]
            y_pred[i, top_k] = 1.0
    else:
        y_pred = y_proba >= 0.5

    return y_pred, meta


def find_thresholds_macro_f1(y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs):
    return find_thresholds_wrapper(y_proba, bin_fmeasure_on_conf_matrix, k, y_true_valid=y_true_valid, y_proba_valid=y_proba_valid, **kwargs) 


def find_thresholds_macro_f1_on_test(y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs):
    return find_thresholds_wrapper(y_proba, bin_fmeasure_on_conf_matrix, k, y_true_valid=y_true, y_proba_valid=y_proba, **kwargs)


def find_thresholds_macro_f1_etu(y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs):
    return find_thresholds_wrapper(y_proba, bin_fmeasure_on_conf_matrix, k, y_true_valid=y_proba, y_proba_valid=y_proba, **kwargs)


def find_thresholds_macro_min_tp_tn(y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs):
    return find_thresholds_wrapper(y_proba, bin_min_tp_tn, k, y_true_valid=y_true_valid, y_proba_valid=y_proba_valid, **kwargs) 


def find_thresholds_macro_min_tp_tn_on_test(y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs):
    return find_thresholds_wrapper(y_proba, bin_min_tp_tn, k, y_true_valid=y_true, y_proba_valid=y_proba, **kwargs) 


def find_thresholds_macro_min_tp_tn_etu(y_proba, k, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs):
    return find_thresholds_wrapper(y_proba, bin_min_tp_tn, k, y_true_valid=y_proba, y_proba_valid=y_proba, **kwargs)