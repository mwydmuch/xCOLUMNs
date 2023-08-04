# simple tests for macro precision at k
# all code uses dense matrices, so it won't scale to really large problems
# first index is always instance, second index is always label

import numpy as np
from scipy.sparse import csr_matrix
from typing import Union

from metrics import macro_precision
from prediction_dense import *
from prediction_sparse import *


def weighted_per_instance(prediction: Union[np.ndarray, csr_matrix], weights: np.ndarray, k: int = 5):
    if isinstance(prediction, np.ndarray):
        # Invoke original implementation of Erik
        return np_weighted_per_instance(prediction, weights, k=k)
    elif isinstance(prediction, csr_matrix):
        # Invoke implementation for sparse matrices
        return csr_weighted_per_instance(prediction, weights, k=k)


def optimal_macro_recall(prediction: np.ndarray, k: int = 5, *, marginals, **kwargs):
    return weighted_per_instance(prediction, 1.0 / (marginals + 0.00001), k=k)


def inv_propensity_weighted_instance(prediction: np.ndarray, k: int = 5, *, inv_ps, **kwargs):
    return weighted_per_instance(prediction, inv_ps, k=k)


def log_weighted_instance(prediction: np.ndarray, k: int = 5, *, marginals, **kwargs):
    weights = -np.log(marginals + 0.0001)
    return weighted_per_instance(prediction, weights, k=k)


def sqrt_weighted_instance(prediction: np.ndarray, k: int = 5, *, marginals, **kwargs):
    weights = 1.0 / np.sqrt(marginals + 0.0001)
    return weighted_per_instance(prediction, weights, k=k)


def power_law_weighted_instance(prediction: np.ndarray, k: int = 5, *, marginals, beta=0.25, **kwargs):
    weights = 1.0 / (marginals + 0.0001)**beta
    return weighted_per_instance(prediction, weights, k=k)


def optimal_instance_precision(prediction: np.ndarray, k: int = 5, **kwargs):
    ni, nl = prediction.shape
    weights = np.ones((nl,), dtype=np.float32)
    return weighted_per_instance(prediction, weights, k=k)


def block_coordinate_coverage(probabilities: np.ndarray, k: int, tolerance: float = 1e-5, max_iter=10, seed: int = None, **kwargs):
    if isinstance(probabilities, np.ndarray):
        # Invoke original implementation of Erik
        return np_block_coordinate_coverage(probabilities, k, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs)
    elif isinstance(probabilities, csr_matrix):
        # Invoke implementation for sparse matrices
        return csr_block_coordinate_coverage(probabilities, k, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs)


def macro_population_cm_risk(probabilities: np.ndarray, k: int, measure_func: callable, 
                             tolerance: float = 1e-5, max_iter=100, seed: int = None, **kwargs):
    if isinstance(probabilities, np.ndarray):
        # Invoke original implementation of Erik
        return np_macro_population_cm_risk(probabilities, k, measure_func, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs)
    elif isinstance(probabilities, csr_matrix):
        # Invoke implementation for sparse matrices
        return csr_macro_population_cm_risk(probabilities, k, measure_func, tolerance=tolerance, max_iter=max_iter, seed=seed, **kwargs)
    

def instance_precision_at_k_on_conf_matrix(tp, fp, fn, k):
    return np.asarray(tp / k).ravel()


def macro_precision_on_conf_matrix(tp, fp, fn, epsilon=1e-5):
    return np.asarray(tp / (tp + fp + epsilon)).ravel()


def macro_recall_on_conf_matrix(tp, fp, fn, epsilon=1e-5):
    return np.asarray(tp / (tp + fn + epsilon)).ravel()


def macro_fmeasure_on_conf_matrix(tp, fp, fn, beta=1.0, epsilon=1e-5):
    precision = macro_precision_on_conf_matrix(tp, fp, fn, epsilon=epsilon)
    recall = macro_recall_on_conf_matrix(tp, fp, fn, epsilon=epsilon)
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)


def block_coordinate_macro_precision(predictions: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return macro_population_cm_risk(predictions, k=k, measure_func=macro_precision_on_conf_matrix, **kwargs)


def block_coordinate_macro_recall(predictions: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return macro_population_cm_risk(predictions, k=k, measure_func=macro_recall_on_conf_matrix, **kwargs)


def block_coordinate_macro_f1(predictions: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return macro_population_cm_risk(predictions, k=k, measure_func=macro_fmeasure_on_conf_matrix, **kwargs)


def block_coordinate_mixed_precision(predictions: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 0.5, **kwargs):
    def mixed_precision_alpha_fn(tp, fp, fn):
        return alpha * instance_precision_at_k_on_conf_matrix(tp, fp, fn, k) + (1 - alpha) * macro_precision_on_conf_matrix(tp, fp, fn)
    return macro_population_cm_risk(predictions, k=k, measure_func=mixed_precision_alpha_fn, **kwargs)


def block_coordinate_instance_prec_macro_f1(predictions: Union[np.ndarray, csr_matrix], k: int = 5, alpha: float = 0.5, **kwargs):
    def mixed_precision_alpha_fn(tp, fp, fn):
        return alpha * instance_precision_at_k_on_conf_matrix(tp, fp, fn, k) + (1 - alpha) * macro_fmeasure_on_conf_matrix(tp, fp, fn)
    return macro_population_cm_risk(predictions, k=k, measure_func=mixed_precision_alpha_fn, **kwargs)
