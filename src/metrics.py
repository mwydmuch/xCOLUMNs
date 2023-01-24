import numpy as np


def precision(*, y_true: np.ndarray, y_pred: np.ndarray, axis: int, epsilon=0.01):
    """
    Given true and predicted labels, calculates the precision along the given axis.
    """
    predicted_positives = np.maximum(np.sum(y_pred, axis=axis), epsilon)
    true_positives = np.sum(y_pred * y_true, axis=axis)
    return true_positives / predicted_positives


def recall(*, y_true: np.ndarray, y_pred: np.ndarray, axis: int, epsilon=0.01):
    """
    Given true and predicted labels, calculates the recall along the given axis.
    """
    positives = np.sum(y_true, axis=axis) + epsilon
    true_positives = np.sum(y_pred * y_true, axis=axis)
    return true_positives / positives


def abandonment(*, y_true: np.ndarray, y_pred: np.ndarray, axis: int):
    """
    Given true and predicted labels, calculates whether there is at least one positive along the given axis.
    """
    return np.greater_equal(np.sum(y_pred * y_true, axis=axis), 1.0).astype(np.float32)


def make_average(fn, **kwargs):
    def macro(y_true: np.ndarray, y_pred: np.ndarray, **inner_kw) -> float:
        return np.mean(fn(y_true=y_true, y_pred=y_pred, **kwargs, **inner_kw))
    return macro


macro_precision = make_average(precision, axis=0)
macro_abandonment = make_average(abandonment, axis=0)
macro_recall = make_average(recall, axis=0)

instance_precision = make_average(precision, axis=1)
instance_abandonment = make_average(abandonment, axis=1)
instance_recall = make_average(recall, axis=1)


__all__ = ["macro_precision", "instance_precision", "macro_abandonment", "instance_abandonment",
           "macro_recall", "instance_recall"]
