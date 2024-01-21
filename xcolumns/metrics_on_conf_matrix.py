import numpy as np
from typing import Union
from numbers import Number


def bin_precision_at_k_on_conf_matrix(tp: Union[Number, np.ndarray], fp: Union[Number, np.ndarray], fn: Union[Number, np.ndarray], tn: Union[Number, np.ndarray], k: int):
    return tp / k


def bin_precision_on_conf_matrix(tp: Union[Number, np.ndarray], fp: Union[Number, np.ndarray], fn: Union[Number, np.ndarray, None], tn: Union[Number, np.ndarray, None], epsilon: float =1e-6):
    return tp / (tp + fp + epsilon)


def bin_recall_on_conf_matrix(tp: Union[Number, np.ndarray], fp: Union[Number, np.ndarray, None], fn: Union[Number, np.ndarray], tn: Union[Number, np.ndarray, None], epsilon: float =1e-6):
    return tp / (tp + fn + epsilon)


def bin_fmeasure_on_conf_matrix(tp: Union[Number, np.ndarray], fp: Union[Number, np.ndarray], fn: Union[Number, np.ndarray], tn: Union[Number, np.ndarray, None], beta: float =1.0, epsilon: float =1e-6):
    return (1 + beta**2) * tp / ((beta**2 * (tp + fp)) + tp + fn + epsilon)

# Alternative definition of F-measure used in some old experiments
# def bin_fmeasure_on_conf_matrix(tp, fp, fn, tn, beta=1.0, epsilon=1e-6):
#     precision = bin_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
#     recall = bin_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
#     return (
#         (1 + beta**2)
#         * precision
#         * recall
#         / (beta**2 * precision + recall + epsilon)
#     )


def macro_precision_on_conf_matrix(tp: np.ndarray, fp: np.ndarray, fn: Union[np.ndarray, None], tn: Union[np.ndarray, None], epsilon: float=1e-6):
    return bin_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon).mean()


def macro_recall_on_conf_matrix(tp: np.ndarray, fp: Union[np.ndarray, None], fn: np.ndarray, tn: Union[np.ndarray, None], epsilon: float=1e-6):
    return bin_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon).mean()


def macro_fmeasure_on_conf_matrix(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: Union[np.ndarray, None], beta: float =1.0, epsilon: float =1e-6):
    return bin_fmeasure_on_conf_matrix(tp, fp, fn, tn, beta=beta, epsilon=epsilon).mean()


def coverage_on_conf_matrix(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: Union[np.ndarray, None]):
    pass