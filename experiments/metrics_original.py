from typing import Union

import numpy as np
from scipy.sparse import csr_matrix


# Original implementation of metrics used in some experiments


def _predicted_positives(  # q in the paper
    y_pred: Union[np.ndarray, csr_matrix], axis: int = None, epsilon: float = 1e-9
):
    """
    Given predicted labels, calculates their number along the given axis.
    """
    return np.asarray(np.maximum(y_pred.mean(axis=axis), epsilon)).ravel()


def _positives(  # p in the paper
    y_true: Union[np.ndarray, csr_matrix], axis: int = None, epsilon: float = 1e-9
):
    """
    Given true labels, calculates their number along the given axis.
    """
    return np.asarray(np.maximum(y_true.mean(axis=axis), epsilon)).ravel()


def _true_positives(  # t in the paper
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    axis: int = None,
):
    """
    Given true and predicted labels, calculates the true positives along the given axis.
    """
    if isinstance(y_true, csr_matrix):
        y_true_x_pred = y_true.multiply(y_pred)
    else:
        y_true_x_pred = y_true * y_pred
    return np.asarray(y_true_x_pred.mean(axis=axis)).ravel()


def _true_negatives(  # t in the paper
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    axis: int = None,
):
    """
    Given true and predicted labels, calculates the true positives along the given axis.
    """
    if isinstance(y_true, csr_matrix):
        y_true_x_pred = y_true.multiply(y_pred)
    else:
        tn = (1 - y_true) * (1 - y_pred)
    return np.asarray(y_true_x_pred.mean(axis=axis)).ravel()


def _precision(
    *,
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    axis: int,
    epsilon: float = 1e-9,
):
    """
    Given true and predicted labels, calculates the precision along the given axis.
    """
    predicted_positives = _predicted_positives(y_pred, axis=axis, epsilon=epsilon)
    true_positives = _true_positives(y_pred, y_true, axis=axis)
    return true_positives / predicted_positives


def _recall(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    axis: int,
    epsilon: float = 1e-9,
):
    """
    Given true and predicted labels, calculates the recall along the given axis.
    """
    positives = _positives(y_true, axis=axis, epsilon=epsilon)
    true_positives = _true_positives(y_pred, y_true, axis=axis)
    return true_positives / positives


def _fmeasure(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    axis: int,
    beta: float = 1,
    epsilon: float = 1e-9,
):
    """
    Given true and predicted labels, calculates the F1 score along the given axis.
    """
    predicted_positives = _predicted_positives(y_pred, axis=axis, epsilon=epsilon)
    true_positives = _true_positives(y_pred, y_true, axis=axis)
    positives = _positives(y_true, axis=axis, epsilon=epsilon)

    return (
        (1 + beta**2) * true_positives / (beta**2 * predicted_positives + positives)
    )

    # Alt
    # precision = true_positives / predicted_positives
    # recall = true_positives / positives

    # return (
    #     (1 + beta**2)
    #     * precision
    #     * recall
    #     / (beta**2 * precision + recall + epsilon)
    # )


def _abandonment(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    axis: int,
):
    """
    Given true and predicted labels, calculates whether there is at least one positive along the given axis.
    """
    return np.greater(_true_positives(y_pred, y_true, axis=axis), 0.0).astype(
        np.float32
    )


def _balanced_accuracy(
    y_true: Union[np.ndarray, csr_matrix],
    y_pred: Union[np.ndarray, csr_matrix],
    axis: int,
):
    """
    Given true and predicted labels, calculates the balanced accuracy along the given axis.
    """
    predicted_positives = _predicted_positives(y_pred, axis=axis)
    true_positives = _true_positives(y_pred, y_true, axis=axis)
    positives = _positives(y_true, axis=axis)

    return (true_positives + positives * (1 - predicted_positives - positives)) / (
        2 * positives * (1 - positives)
    )


def make_average(fn, **kwargs):
    def avg_func(
        y_true: Union[np.ndarray, csr_matrix],
        y_pred: Union[np.ndarray, csr_matrix],
        **inner_kw,
    ) -> float:
        return fn(y_true=y_true, y_pred=y_pred, **kwargs, **inner_kw).mean()

    return avg_func


macro_precision_v0 = make_average(_precision, axis=0)
macro_recall_v0 = make_average(_recall, axis=0)
macro_f1_v0 = make_average(_fmeasure, axis=0)
macro_abandonment_v0 = make_average(_abandonment, axis=0)
macro_balanced_accuracy_v0 = make_average(_balanced_accuracy, axis=0)

instance_precision_v0 = make_average(_precision, axis=1)
instance_recall_v0 = make_average(_recall, axis=1)
instance_f1_v0 = make_average(_fmeasure, axis=1)
instance_abandonment_v0 = make_average(_abandonment, axis=1)
instance_balanced_accuracy_v0 = make_average(_balanced_accuracy, axis=1)
