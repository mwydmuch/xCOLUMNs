from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .confusion_matrix import calculate_confusion_matrix
from .types import *


########################################################################################
# Helper functions
########################################################################################


def check_y_pred(y_pred: Matrix, k: int = None) -> None:
    if (isinstance(y_pred, DenseMatrix) and ((y_pred == 0) & (y_pred == 1)).any()) or (
        isinstance(y_pred, csr_matrix)
        and ((y_pred.data == 0) & (y_pred.data == 1)).any()
    ):
        raise ValueError("y_pred must be a binary matrix")

    if k is not None and k > 0 and (y_pred.sum(axis=1) != k).any():
        raise ValueError("y_pred must have exactly k positive labels per instance")


########################################################################################
# Functional implementations of binary metrics (without any safe checks)
########################################################################################


def bin_precision_at_k_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray, None],
    fn: Union[Number, np.ndarray, None],
    tn: Union[Number, np.ndarray, None],
    k: int,
) -> Union[Number, np.ndarray]:
    return tp / k


def bin_precision_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray, None],
    tn: Union[Number, np.ndarray, None],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate precision from the given true positives, false positives, false negatives and true negatives.
    """
    return tp / (tp + fp + epsilon)


def bin_recall_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray, None],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray, None],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate recall from the given true positives, false positives, false negatives and true negatives.
    """
    return tp / (tp + fn + epsilon)


def bin_fmeasure_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray, None],
    beta: float = 1.0,
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate F-measure from the given true positives, false positives, false negatives and true negatives.
    """
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


def bin_ballanced_accuracy_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate ballanced accuracy from the given true positives, false positives, false negatives and true negatives.
    """
    tpr = tp / (tp + fn + epsilon)
    tnr = tn / (tn + fp + epsilon)
    return (tpr + tnr) / 2


def bin_h_mean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate H-mean (harmonic mean) from the given true positives, false positives, false negatives and true negatives.
    """
    tpr = tp / (tp + fn + epsilon)
    tnr = tn / (tn + fp + epsilon)
    return (2 * tpr * tnr) / (tpr + tnr + 1e-6)


def bin_g_mean_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate G-mean (geometric mean) from the given true positives, false positives, false negatives and true negatives.
    """
    tpr = tp / (tp + fn + epsilon)
    tnr = tn / (tn + fp + epsilon)
    return (tpr * tnr) ** 0.5


########################################################################################
# Functional implementations of multilabel metrics
########################################################################################


def macro_precision_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: Union[np.ndarray, None],
    tn: Union[np.ndarray, None],
    epsilon: float = 1e-6,
) -> Number:
    return bin_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon).mean()


def macro_recall_on_conf_matrix(
    tp: np.ndarray,
    fp: Union[np.ndarray, None],
    fn: np.ndarray,
    tn: Union[np.ndarray, None],
    epsilon: float = 1e-6,
) -> Number:
    return bin_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon).mean()


def micro_fmeasure_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: Union[np.ndarray, None],
    beta: float = 1.0,
    epsilon: float = 1e-6,
) -> Number:
    return bin_fmeasure_on_conf_matrix(
        tp.sum(), fp.sum(), fn.sum(), None, beta=beta, epsilon=epsilon
    )


def macro_fmeasure_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: Union[np.ndarray, None],
    beta: float = 1.0,
    epsilon: float = 1e-6,
) -> Number:
    return bin_fmeasure_on_conf_matrix(
        tp, fp, fn, tn, beta=beta, epsilon=epsilon
    ).mean()


def coverage_on_conf_matrix(
    tp: np.ndarray,
    fp: Union[np.ndarray, None],
    fn: Union[np.ndarray, None],
    tn: Union[np.ndarray, None],
) -> Number:
    return (tp > 0).mean()


def hamming_score_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
    normalize: bool = True,
) -> Number:
    score = tp + tn
    if normalize:
        score /= tp + fp + fn + tn
    return score.sum()


def hamming_loss_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
    normalize: bool = True,
) -> Number:
    loss = fp + fn
    if normalize:
        loss /= tp + fp + fn + tn
    return loss.sum()


def hamming_score(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    normalize: bool = True,
) -> Number:
    """
    Calculate hamming score for the given true and predicted labels.
    """
    return hamming_score_on_conf_matrix(
        *calculate_confusion_matrix(y_true, y_pred, normalize=False),
        normalize=normalize,
    )


def hamming_loss(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    normalize: bool = True,
) -> Number:
    """
    Calculate hamming loss for the given true and predicted labels.
    """
    return hamming_loss_on_conf_matrix(
        *calculate_confusion_matrix(y_true, y_pred, normalize=False),
        normalize=normalize,
    )


def micro_fmeasure(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    beta: float = 1.0,
    epsilon: float = 1e-6,
) -> Number:
    """
    Calculate micro-averaged F-measure for the given true and predicted labels.
    """
    return micro_fmeasure_on_conf_matrix(
        *calculate_confusion_matrix(y_true, y_pred, normalize=False),
        beta=beta,
        epsilon=epsilon,
    )


def micro_f1(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    epsilon: float = 1e-6,
) -> Number:
    """
    Alias for micro_fmeasure with beta=1.0.
    """
    return micro_fmeasure(y_true, y_pred, beta=1.0, epsilon=epsilon)


# class MetricsCollection(object):
#     def __init__(self, metrics: Union[Dict[str, Metric], List[Metric]):
#         if isinstance(metrics, Iter):
#             metrics = {metric.name: metric for metric in metrics}


#     def calculate(self, y_true, y_pred):
#         tp = fp = fn = tn = None
#         for metric in self.metrics:
#             if isinstance(metric, MetricOnConfusionMatrix):
#                 if tp is None:
#                     tp, fp, fn, tn = metric.calculate_confusion_matrix(y_true, y_pred)


#             metric.calculate(y_true, y_pred)

#         metrics_on_conf_matrix =


########################################################################################
# Implementations of multi-label metrics as objects
########################################################################################


class Metric:
    """
    Abstract class for all metrics.
    """

    def __init__(self, name, k=None):
        if k is not None and k > 0:
            self.name = f"{name}@{k}"
        else:
            self.name = name
        self.k = k

    def calculate(self, y_true, y_pred):
        raise NotImplementedError()

    def _check_pred(self, y_pred):
        y_pred.sum(axis=1) == self.k


class MetricOnConfusionMatrix(Metric):
    """
    Abstract class for mtrics defined on confusion matrix (tp, fp, fn, tn)
    """

    def __init__(self, name, k=None, needs_tn=True):
        super().__init__(name, k=k)
        self.needs_tn = needs_tn

    def calculate_on_confusion_matrix(self, tp, fp, fn, tn):
        raise NotImplementedError()

    def torch_calculate_on_confusion_matrix(self, tp, fp, fn, tn):
        return self.calculate_on_confusion_matrix(tp, fp, fn, tn)

    def calculate(self, y_true, y_pred):
        tp, fp, fn, tn = calculate_confusion_matrix(
            y_true, y_pred, normalize=True, skip_tn=not self.needs_tn
        )
        return self.calculate_on_confusion_matrix(tp, fp, fn, tn)


class MetricLinearyDecomposableOverLabels(MetricOnConfusionMatrix):
    def __init__(self, name, k=None, needs_tn=True):
        super().__init__(name, k=k, needs_tn=needs_tn)

    def calculate_per_label_values_on_confusion_matrix(self, tp, fp, fn, tn):
        raise NotImplementedError()

    def calculate_per_label_values(self, y_true, y_pred):
        tp, fp, fn, tn = calculate_confusion_matrix(y_true, y_pred)
        return self.calculate_per_label_values_on_confusion_matrix(tp, fp, fn, tn)

    def calculate_on_confusion_matrix(self, tp, fp, fn, tn):
        return self.calculate_per_label_values_on_confusion_matrix(tp, fp, fn, tn).sum()


# class MacroAveragedMetric(MetricLinearyDecomposableOverLabels):
#     def __init__(self, name, k=None, needs_tn=True):
#         super().__init__(name, k=k, needs_tn=needs_tn)

#     def calculate_per_label_values(self, y_true, y_pred):
#         tp, fp, fn, tn = calculate_confusion_matrix(y_true, y_pred)
#         return self.calculate_per_label_values_on_confusion_matrix(tp, fp, fn, tn)

#     def calculate_on_confusion_matrix(self, tp, fp, fn, tn):
#         return self.calculate_per_label_values_on_confusion_matrix(tp, fp, fn, tn).sum()


class MacroFMeasure(MetricLinearyDecomposableOverLabels):
    def __init__(self, k=0, beta=1.0, epsilon=1e-6):
        if beta == 1.0:
            super().__init__(f"macro-f1-measure", k=k, needs_tn=False)
        else:
            super().__init__(f"macro-f_beta={beta}-measure", k=k, needs_tn=False)
        self.beta = beta
        self.epsilon = epsilon

    def calculate_binary_values(self, tp, fp, fn, tn):
        return (
            bin_fmeasure_on_conf_matrix(
                tp, fp, fn, tn, beta=self.beta, epsilon=self.epsilon
            )
            / tp.shape[0]
        )


class MicroFMeasure(MetricOnConfusionMatrix):
    def __init__(self, k=0, beta=1.0, epsilon=1e-6):
        if beta == 1.0:
            super().__init__(f"micro-f1-measure", k=k, needs_tn=False)
        else:
            super().__init__(f"micro-f_beta={beta}-measure", k=k, needs_tn=False)
        self.beta = beta
        self.epsilon = epsilon

    def calculate_on_confusion_matrix(self, tp, fp, fn, tn):
        return micro_fmeasure_on_conf_matrix(
            tp, fp, fn, tn, beta=self.beta, epsilon=self.epsilon
        )
