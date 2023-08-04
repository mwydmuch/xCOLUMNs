import numpy as np
from scipy.sparse import csr_matrix
from typing import Union


def _predicted_positives(y_pred: Union[np.ndarray, csr_matrix], axis: int=None, epsilon: float=1e-5):
    """
    Given predicted labels, calculates their number along the given axis.
    """
    return np.asarray(np.maximum(y_pred.sum(axis=axis), epsilon)).ravel()


def _positives(y_true: Union[np.ndarray, csr_matrix], axis: int=None, epsilon: float=1e-5):
    """
    Given true labels, calculates their number along the given axis.
    """
    return np.asarray(y_true.sum(axis=axis)).ravel() + epsilon


def _true_positives(y_true: Union[np.ndarray, csr_matrix], y_pred: Union[np.ndarray, csr_matrix], axis: int=None):
    """
    Given true and predicted labels, calculates the true positives along the given axis.
    """
    if isinstance(y_true, csr_matrix):
        y_true_x_pred = y_true.multiply(y_pred)
    else:
        y_true_x_pred = y_true * y_pred
    return np.asarray(y_true_x_pred.sum(axis=axis)).ravel()


def precision(*, y_true: Union[np.ndarray, csr_matrix], y_pred: Union[np.ndarray, csr_matrix], axis: int, epsilon: float=1e-5):
    """
    Given true and predicted labels, calculates the precision along the given axis.
    """
    predicted_positives = _predicted_positives(y_pred, axis=axis, epsilon=epsilon)
    true_positives = _true_positives(y_pred, y_true, axis=axis)
    return true_positives / predicted_positives


def recall(*, y_true: Union[np.ndarray, csr_matrix], y_pred: Union[np.ndarray, csr_matrix], axis: int, epsilon: float=1e-5):
    """
    Given true and predicted labels, calculates the recall along the given axis.
    """
    positives = _positives(y_true, axis=axis, epsilon=epsilon)
    true_positives = _true_positives(y_pred, y_true, axis=axis)
    return true_positives / positives


def fmeasure(*, y_true: Union[np.ndarray, csr_matrix], y_pred: Union[np.ndarray, csr_matrix], axis: int, beta: float=1, epsilon: float=1e-5):
    """
    Given true and predicted labels, calculates the F1 score along the given axis.
    """
    predicted_positives = _predicted_positives(y_pred, axis=axis, epsilon=epsilon)
    true_positives = _true_positives(y_pred, y_true, axis=axis)
    precision = true_positives / predicted_positives

    positives = _positives(y_true, axis=axis, epsilon=epsilon)
    recall = true_positives / positives

    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)


def abandonment(*, y_true: Union[np.ndarray, csr_matrix], y_pred: Union[np.ndarray, csr_matrix], axis: int):
    """
    Given true and predicted labels, calculates whether there is at least one positive along the given axis.
    """
    return np.greater_equal(_true_positives(y_pred, y_true, axis=axis), 1.0).astype(np.float32)


def make_average(fn, **kwargs):
    def avg_fun(y_true: Union[np.ndarray, csr_matrix], y_pred: Union[np.ndarray, csr_matrix], **inner_kw) -> float:
        return fn(y_true=y_true, y_pred=y_pred, **kwargs, **inner_kw).mean()
    return avg_fun


macro_precision = make_average(precision, axis=0)
macro_recall = make_average(recall, axis=0)
macro_f1 = make_average(fmeasure, axis=0)
macro_abandonment = make_average(abandonment, axis=0)

instance_precision = make_average(precision, axis=1)
instance_recall = make_average(recall, axis=1)
instance_f1 = make_average(fmeasure, axis=1)
instance_abandonment = make_average(abandonment, axis=1)


__all__ = ["macro_precision", "instance_precision", "macro_abandonment", "instance_abandonment",
           "macro_recall", "instance_recall", "macro_f1", "instance_f1"]
