import inspect
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .confusion_matrix import calculate_confusion_matrix
from .types import DenseMatrix, Matrix, Number


########################################################################################
# Helper functions
########################################################################################


def check_if_y_pred_at_k(y_pred: Matrix, k: int) -> None:
    """ """

    if (isinstance(y_pred, DenseMatrix) and ((y_pred == 0) & (y_pred == 1)).any()) or (
        isinstance(y_pred, csr_matrix)
        and ((y_pred.data == 0) & (y_pred.data == 1)).any()
    ):
        raise ValueError("y_pred must be a binary matrix")

    if k is not None and k > 0 and (y_pred.sum(axis=1) != k).any():
        raise ValueError("y_pred must have exactly k positive labels per instance")


def _add_kwargs_to_signature(func_org: Callable, func_new) -> Callable:
    sig_org = inspect.signature(func_org)
    sig_new = inspect.signature(func_new)
    func_new.__signature__ = sig_new.replace(
        parameters=[
            p
            for p in sig_new.parameters.values()
            if p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        + [
            p
            for p in sig_org.parameters.values()
            if p.default != inspect.Parameter.empty
        ]
    )
    return func_new


def _make_macro_metric_on_conf_mat(binary_metric, name) -> Callable:
    def macro_metric_on_conf_matrix(
        tp: DenseMatrix,
        fp: DenseMatrix,
        fn: DenseMatrix,
        tn: DenseMatrix,
    ) -> Number:
        return binary_metric(tp, fp, fn, tn).mean()

    macro_metric_on_conf_matrix.__doc__ = f"""
    Calculate macro-averaged {name} from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    See :meth:`{binary_metric.__name__}` for definition.
    """

    return _add_kwargs_to_signature(binary_metric, macro_metric_on_conf_matrix)


def _make_micro_metric_on_conf_mat(binary_metric, name) -> Callable:
    def micro_metric_on_conf_matrix(
        tp: DenseMatrix,
        fp: DenseMatrix,
        fn: DenseMatrix,
        tn: DenseMatrix,
        **kwargs,
    ) -> Number:
        return binary_metric(tp.sum(), fp.sum(), fn.sum(), tn.sum(), **kwargs)

    micro_metric_on_conf_matrix.__doc__ = f"""
    Calculate macro-averaged {name} from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    See :meth:`{binary_metric.__name__}` for definition.
    """

    return _add_kwargs_to_signature(binary_metric, micro_metric_on_conf_matrix)


def _make_metric_on_y_true_and_y_pred(
    metric_on_conf_matrix, name, skip_tn=False
) -> Callable:
    def metric_on_y_true_and_y_pred(
        y_true: Matrix,
        y_pred: Matrix,
        **kwargs,
    ):
        C = calculate_confusion_matrix(y_true, y_pred, normalize=True, skip_tn=skip_tn)
        return metric_on_conf_matrix(*C, **kwargs)

    metric_on_y_true_and_y_pred.__doc__ = f"""
    Calculate {name} matric from the given true and predicted labels.

    See :meth:`{metric_on_conf_matrix.__name__}` for definition.
    """

    return _add_kwargs_to_signature(metric_on_conf_matrix, metric_on_y_true_and_y_pred)


########################################################################################
# Accuracy / 0-1 loss / Hamming score / loss
########################################################################################


def binary_accuracy_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
    normalize: bool = True,
) -> Union[Number, np.ndarray]:
    acc = tp + tn
    if normalize:
        acc /= tp + fp + fn + tn
    return acc


def binary_0_1_loss_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray],
    normalize: bool = True,
) -> Union[Number, np.ndarray]:
    loss = fp + fn
    if normalize:
        loss /= tp + fp + fn + tn
    return loss


def hamming_score_on_conf_matrix(
    tp: DenseMatrix,
    fp: DenseMatrix,
    fn: DenseMatrix,
    tn: DenseMatrix,
    normalize: bool = True,
) -> Number:
    return binary_accuracy_on_conf_matrix(tp, fp, fn, tn, normalize=normalize).mean()


def hamming_loss_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
    normalize: bool = True,
) -> Number:
    return binary_0_1_loss_on_conf_matrix(tp, fp, fn, tn, normalize=normalize).mean()


binary_accuracy = _make_metric_on_y_true_and_y_pred(
    binary_accuracy_on_conf_matrix, "accuracy"
)
binary_0_1_loss = _make_metric_on_y_true_and_y_pred(
    binary_0_1_loss_on_conf_matrix, "0/1 loss"
)
hamming_score = _make_metric_on_y_true_and_y_pred(
    hamming_score_on_conf_matrix, "Hamming score"
)
hamming_loss = _make_metric_on_y_true_and_y_pred(
    hamming_loss_on_conf_matrix, "Hamming loss"
)


########################################################################################
# Precision at k
########################################################################################


def binary_precision_at_k_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray, None],
    fn: Union[Number, np.ndarray, None],
    tn: Union[Number, np.ndarray, None],
    k: int,
) -> Union[Number, np.ndarray]:
    return tp / k


def precision_at_k_on_conf_matrix(
    tp: np.ndarray,
    fp: Union[np.ndarray, None],
    fn: Union[np.ndarray, None],
    tn: Union[np.ndarray, None],
    k: int,
) -> Number:
    return binary_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k).sum()


precision_at_k = _make_metric_on_y_true_and_y_pred(
    precision_at_k_on_conf_matrix, "precision at k", skip_tn=True
)


########################################################################################
# Precision
########################################################################################


def binary_precision_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray, None],
    tn: Union[Number, np.ndarray, None],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate binary precision
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    .. math::
        \\text{precision} = \\frac{TP}{TP + FP + \\epsilon}

    where :math:`\\epsilon` is a very small number to avoid division by zero.
    """
    return tp / (tp + fp + epsilon)


macro_precision_on_conf_matrix = _make_macro_metric_on_conf_mat(
    binary_precision_on_conf_matrix, "precision"
)
micro_precision_on_conf_matrix = _make_micro_metric_on_conf_mat(
    binary_precision_on_conf_matrix, "precision"
)
binary_precision = _make_metric_on_y_true_and_y_pred(
    binary_precision_on_conf_matrix, "binary precision", skip_tn=True
)
macro_precision = _make_metric_on_y_true_and_y_pred(
    macro_precision_on_conf_matrix, "macro-averaged precision", skip_tn=True
)
micro_precision = _make_metric_on_y_true_and_y_pred(
    micro_precision_on_conf_matrix, "micro-averaged precision", skip_tn=True
)


########################################################################################
# Recall
########################################################################################


def binary_recall_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray, None],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray, None],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate binary recall
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives:

    .. math::
        \\text{recall} = \\frac{TP}{TP + FN + \\epsilon}

    where :math:`\\epsilon` is a very small number to avoid division by zero.
    """
    return tp / (tp + fn + epsilon)


macro_recall_on_conf_matrix = _make_macro_metric_on_conf_mat(
    binary_recall_on_conf_matrix, "recall"
)
micro_recall_on_conf_matrix = _make_micro_metric_on_conf_mat(
    binary_recall_on_conf_matrix, "recall"
)
binary_recall = _make_metric_on_y_true_and_y_pred(
    binary_recall_on_conf_matrix, "binary recall", skip_tn=True
)
macro_recall = _make_metric_on_y_true_and_y_pred(
    macro_recall_on_conf_matrix, "macro-averaged recall", skip_tn=True
)
micro_recall = _make_metric_on_y_true_and_y_pred(
    micro_recall_on_conf_matrix, "micro-averaged recall", skip_tn=True
)


########################################################################################
# F-score
########################################################################################


def binary_fbeta_score_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray, None],
    beta: float = 1.0,
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Compute the binary F-beta score.
    from the given true positives, false positives, false negatives and true negatives.
    """
    return (1 + beta**2) * tp / ((beta**2 * (tp + fp)) + tp + fn + epsilon)


# Alternative definition of F-beta score used in some old experiments
# def binary_fbeta_score_on_conf_matrix(tp, fp, fn, tn, beta=1.0, epsilon=1e-6):
#     precision = binary_precision_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
#     recall = binary_recall_on_conf_matrix(tp, fp, fn, tn, epsilon=epsilon)
#     return (
#         (1 + beta**2)
#         * precision
#         * recall
#         / (beta**2 * precision + recall + epsilon)
#     )


def binary_f1_score_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray, None],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate binary F1 score, also known as balanced F-score or F-measure
    from the given true positives, false positives, false negatives and true negatives.
    This is an alias for binary_fbeta_score_on_conf_matrix with beta=1.0.
    """
    return binary_fbeta_score_on_conf_matrix(tp, fp, fn, tn, beta=1.0, epsilon=epsilon)


macro_fbeta_score_on_conf_matrix = _make_macro_metric_on_conf_mat(
    binary_fbeta_score_on_conf_matrix, "F-beta score"
)
micro_fbeta_score_on_conf_matrix = _make_micro_metric_on_conf_mat(
    binary_fbeta_score_on_conf_matrix, "F-beta score"
)
binary_fbeta_score = _make_metric_on_y_true_and_y_pred(
    binary_fbeta_score_on_conf_matrix, "binary F-beta score", skip_tn=True
)
macro_fbeta_score = _make_metric_on_y_true_and_y_pred(
    macro_fbeta_score_on_conf_matrix, "macro-averaged F-beta score", skip_tn=True
)
micro_fbeta_score = _make_metric_on_y_true_and_y_pred(
    micro_fbeta_score_on_conf_matrix, "micro-averaged F-beta score", skip_tn=True
)

macro_f1_score_on_conf_matrix = _make_macro_metric_on_conf_mat(
    binary_fbeta_score_on_conf_matrix, "F1 score"
)
micro_f1_score_on_conf_matrix = _make_micro_metric_on_conf_mat(
    binary_fbeta_score_on_conf_matrix, "F1 score"
)
binary_f1_score = _make_metric_on_y_true_and_y_pred(
    binary_fbeta_score_on_conf_matrix, "binary F1 score"
)
macro_f1_score = _make_metric_on_y_true_and_y_pred(
    macro_fbeta_score_on_conf_matrix, "macro-averaged F1 score"
)
micro_f1_score = _make_metric_on_y_true_and_y_pred(
    micro_fbeta_score_on_conf_matrix, "micro-averaged F1 score"
)


########################################################################################
# Jaccard score
########################################################################################


def binary_jaccard_score_on_conf_matrix(
    tp: Union[Number, np.ndarray],
    fp: Union[Number, np.ndarray],
    fn: Union[Number, np.ndarray],
    tn: Union[Number, np.ndarray, None],
    epsilon: float = 1e-6,
) -> Union[Number, np.ndarray]:
    """
    Calculate Jaccard score
    from the given true positives, false positives, false negatives and true negatives.
    """
    return tp / (tp + fp + fn + epsilon)


macro_jaccard_score_on_conf_matrix = _make_macro_metric_on_conf_mat(
    binary_jaccard_score_on_conf_matrix, "Jaccard score"
)
micro_jaccard_score_on_conf_matrix = _make_micro_metric_on_conf_mat(
    binary_jaccard_score_on_conf_matrix, "Jaccard score"
)
binary_jaccard_score = _make_metric_on_y_true_and_y_pred(
    binary_jaccard_score_on_conf_matrix, "binary Jaccard score", skip_tn=True
)
macro_jaccard_score = _make_metric_on_y_true_and_y_pred(
    macro_jaccard_score_on_conf_matrix, "macro-averaged Jaccard score", skip_tn=True
)
micro_jaccard_score = _make_metric_on_y_true_and_y_pred(
    micro_jaccard_score_on_conf_matrix, "micro-averaged Jaccard score", skip_tn=True
)


########################################################################################
# Ballanced accuracy
########################################################################################


def binary_balanced_accuracy_on_conf_matrix(
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


macro_balanced_accuracy_on_conf_matrix = _make_macro_metric_on_conf_mat(
    binary_balanced_accuracy_on_conf_matrix, "balanced accuracy"
)
micro_balanced_accuracy_on_conf_matrix = _make_micro_metric_on_conf_mat(
    binary_balanced_accuracy_on_conf_matrix, "balanced accuracy"
)
binary_balanced_accuracy = _make_metric_on_y_true_and_y_pred(
    binary_balanced_accuracy_on_conf_matrix, "binary balanced accuracy"
)
macro_balanced_accuracy = _make_metric_on_y_true_and_y_pred(
    macro_balanced_accuracy_on_conf_matrix, "macro-averaged balanced accuracy"
)
micro_balanced_accuracy = _make_metric_on_y_true_and_y_pred(
    micro_balanced_accuracy_on_conf_matrix, "micro-averaged balanced accuracy"
)


########################################################################################
# G-mean
########################################################################################


def binary_gmean_on_conf_matrix(
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


macro_gmean_on_conf_matrix = _make_macro_metric_on_conf_mat(
    binary_gmean_on_conf_matrix, "G-mean"
)
micro_gmean_on_conf_matrix = _make_micro_metric_on_conf_mat(
    binary_gmean_on_conf_matrix, "G-mean"
)
binary_gmean = _make_metric_on_y_true_and_y_pred(
    binary_gmean_on_conf_matrix, "binary G-mean"
)
macro_gmean = _make_metric_on_y_true_and_y_pred(
    macro_gmean_on_conf_matrix, "macro-averaged G-mean"
)
micro_gmean = _make_metric_on_y_true_and_y_pred(
    micro_gmean_on_conf_matrix, "micro-averaged G-mean"
)


########################################################################################
# H-mean
########################################################################################


def binary_hmean_on_conf_matrix(
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


macro_hmean_on_conf_matrix = _make_macro_metric_on_conf_mat(
    binary_hmean_on_conf_matrix, "H-mean"
)
micro_hmean_on_conf_matrix = _make_micro_metric_on_conf_mat(
    binary_hmean_on_conf_matrix, "H-mean"
)
binary_hmean = _make_metric_on_y_true_and_y_pred(
    binary_hmean_on_conf_matrix, "binary H-mean"
)
macro_hmean = _make_metric_on_y_true_and_y_pred(
    macro_hmean_on_conf_matrix, "macro-averaged H-mean"
)
micro_hmean = _make_metric_on_y_true_and_y_pred(
    micro_hmean_on_conf_matrix, "micro-averaged H-mean"
)


########################################################################################
# Coverage
########################################################################################


def coverage_on_conf_matrix(
    tp: np.ndarray,
    fp: Union[np.ndarray, None],
    fn: Union[np.ndarray, None],
    tn: Union[np.ndarray, None],
) -> Number:
    return (tp > 0).mean()


coverage = _make_metric_on_y_true_and_y_pred(
    coverage_on_conf_matrix, "coverage", skip_tn=True
)
