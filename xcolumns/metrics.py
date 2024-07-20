from math import log
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .confusion_matrix import calculate_confusion_matrix
from .types import DenseMatrix, Matrix, Number
from .utils import add_kwargs_to_signature


########################################################################################
# Helper functions
########################################################################################


def check_if_y_pred_at_k(y_pred: Matrix, k: int) -> bool:
    """
    Checks if y_pred is a binary matrix with exactly k positive labels per instance.

    Args:
        y_pred: The binary labels matrix.
        k: Number of positive labels required per instance.

    Returns:
        True if y_pred is a binary matrix with exactly k positive labels per instance.
    """

    if (isinstance(y_pred, DenseMatrix) and ((y_pred == 0) & (y_pred == 1)).any()) or (
        isinstance(y_pred, csr_matrix)
        and ((y_pred.data == 0) & (y_pred.data == 1)).any()
    ):
        raise ValueError("y_pred must be a binary matrix")

    return k > 0 and (y_pred.sum(axis=1) == k).all()


def make_macro_metric_on_conf_matrix(binary_metric, name) -> Callable:
    """
    This is a factory function to create a macro-averaged metric
    from a binary metric defined on confusion matrix: binary_metric(tp, fp, fn, tn).

    Args:
        binary_metric: binary metric defined on confusion matrix
        name: name of the metric to be used in the docstring

    Returns:
        The function calculating the macro-averaged metric defined on a confusion matrix.
    """

    def macro_metric_on_conf_matrix(
        tp: DenseMatrix,
        fp: DenseMatrix,
        fn: DenseMatrix,
        tn: DenseMatrix,
        **kwargs,
    ) -> Number:
        return binary_metric(tp, fp, fn, tn, **kwargs).mean()

    macro_metric_on_conf_matrix.__doc__ = f"""
    Calculates macro-averaged {name} from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    See :func:`{binary_metric.__name__}` for definition.
    """

    return add_kwargs_to_signature(macro_metric_on_conf_matrix, binary_metric)


def make_micro_metric_on_conf_matrix(
    binary_metric: Callable, metric_name: str
) -> Callable:
    """
    This is a factory function to create a micro-averaged metric
    from a binary metric defined on confusion matrix: binary_metric(tp, fp, fn, tn).

    Args:
        binary_metric: binary metric defined on confusion matrix
        name: name of the metric to be used in the docstring

    Returns:
        The function calculating the micro-averaged metric defined on a confusion matrix.
    """

    def micro_metric_on_conf_matrix(
        tp: DenseMatrix,
        fp: DenseMatrix,
        fn: DenseMatrix,
        tn: DenseMatrix,
        **kwargs,
    ) -> Number:
        return binary_metric(tp.sum(), fp.sum(), fn.sum(), tn.sum(), **kwargs)

    micro_metric_on_conf_matrix.__doc__ = f"""
    Calculates macro-averaged {metric_name} from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    See :func:`{binary_metric.__name__}` for definition.
    """

    return add_kwargs_to_signature(micro_metric_on_conf_matrix, binary_metric)


def make_metric_on_y_true_and_y_pred(
    metric_on_conf_matrix: Callable, metric_name: str, skip_tn: bool = False
) -> Callable:
    """
    This is a factory function to create
    a metric calculated on true and predicted labels: metric(y_true, y_pred)
    from metric calculated on confusion matrix: metric(tp, fp, fn, tn).

    Args:
        metric_on_conf_matrix: metric calculated on confusion matrix
        metric_name: name of the metric to be used in the docstring
        skip_tn: whether to skip true negatives

    Returns:
        Function etric calculated on true and predicted labels
    """

    def metric_on_y_true_and_y_pred(
        y_true: Matrix,
        y_pred: Matrix,
        **kwargs,
    ):
        C = calculate_confusion_matrix(y_true, y_pred, normalize=True, skip_tn=skip_tn)
        return metric_on_conf_matrix(*C, **kwargs)

    metric_on_y_true_and_y_pred.__doc__ = f"""
    Calculates {metric_name} matric from the given true and predicted labels.

    See :func:`{metric_on_conf_matrix.__name__}` for definition.
    """

    return add_kwargs_to_signature(metric_on_y_true_and_y_pred, metric_on_conf_matrix)


########################################################################################
# Label counts / priors / propensities
########################################################################################


def label_counts(y: Matrix) -> DenseMatrix:
    """
    Counts number of occurrences of each label.

    Args:
        y: The binary labels matrix.

    Returns:
        An vector the count of label occurrences.
    """
    if len(y.shape) > 2:
        raise ValueError("y must be a binary matrix")

    counts = y.sum(axis=0)

    return counts


def label_priors(y: Matrix) -> DenseMatrix:
    """
    Calculates prior probablity of each label.

    Args:
        y: The binary labels matrix.

    Returns:
        An vector the label priors.
    """
    counts = label_counts(y)
    priors = counts / y.shape[0]

    return priors


def inverse_label_priors(y: Matrix) -> DenseMatrix:
    """
    Calculates inverse of prior probablity of each label.

    Args:
        y: The binary labels matrix.

    Returns:
        An vector the inverse label priors.
    """
    return 1.0 / label_priors(y)


def jpv_propensities(y: Matrix, a: float = 0.55, b: float = 1.5) -> DenseMatrix:
    """
    Calculates propensity of labels using an empirical model proposed in Jain et al. 2016.
    Propensity :math:`p_j` of label :math:`j` is calculated as:

    .. math::

        c = (\\log n - 1)(b + 1)^a \\,, \\
        p_j = \\frac{1}{1 + c(n_l + b)^{-a}} \\,,

    where :math:`n` is total number of data points, :math:`n_j` is total number of data points for
    and :math:`a` and :math:`b` are dataset specific parameters.

    Args:
        y: The binary labels matrix.
        a: A dataset specific parameter, typical values:
            - 0.5: ``WikiLSHTC-325K`` and ``WikipediaLarge-500K``
            - 0.6: ``Amazon-670K`` and ``Amazon-3M``
            - 0.55: otherwise
        b: Dataset specific parameter, typical values:
            - 0.4: ``WikiLSHTC-325K`` and ``WikipediaLarge-500K``
            - 2.6: ``Amazon-670K`` and ``Amazon-3M``
            - 1.5: otherwise

    Returns:
        An vector the label propensities.
    """
    return 1.0 / jpv_inverse_propensities(y, a, b)


def jpv_inverse_propensities(y: Matrix, a: float = 0.55, b: float = 1.5) -> DenseMatrix:
    """
    Calculate inverse propensities of labels an empirical model proposed in Jain et al. 2016.
    Inverse propensity :math:`q_j` of label :math:`j` is calculated as:

    .. math::

        c = (\\log n - 1)(b + 1)^a \\,, \\
        q_j = 1 + c(n_j + b)^{-A} \\,,

    where :math:`n` is total number of data points, :math:`n_j` is total number of data points for
    and :math:`a` and :math:`b` are dataset specific parameters.

    Args:
        y: The binary labels matrix.
        a: A dataset specific parameter, typical values:
            - 0.5: ``WikiLSHTC-325K`` and ``WikipediaLarge-500K``
            - 0.6: ``Amazon-670K`` and ``Amazon-3M``
            - 0.55: otherwise
        b: Dataset specific parameter, typical values:
            - 0.4: ``WikiLSHTC-325K`` and ``WikipediaLarge-500K``
            - 2.6: ``Amazon-670K`` and ``Amazon-3M``
            - 1.5: otherwise

    Returns:
        An vector the inverse label propensities.
    """
    n = y.shape[0]
    counts = label_counts(y)
    C = (log(n) - 1) * (b + 1) ** a
    inv_ps = 1 + C * (counts + b) ** -a
    return inv_ps


########################################################################################
# Accuracy / 0-1 loss / Hamming score / loss
########################################################################################


def binary_accuracy_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix],
    normalize: bool = True,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates binary accuracy from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    .. math::
        \text{accuracy} = \frac{TP + TN}{TP + FP + FN + TN}

    If normalize is False, returns the sum of true positives and true negatives.
    """
    acc = tp + tn
    if normalize:
        acc /= tp + fp + fn + tn
    return acc


def binary_0_1_loss_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix],
    normalize: bool = True,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates binary 0/1 from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    .. math::
        \text{accuracy} = \frac{FP + FN}{TP + FP + FN + TN}

    If normalize is False, returns the sum of false positives and false negatives.
    """
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
    """
    Calculates Hamming score from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    Hamming score is the fraction of correctly predicted labels.
    If normalize is False, returns the mean of true positives and true negatives across all labels.
    """
    return binary_accuracy_on_conf_matrix(tp, fp, fn, tn, normalize=normalize).mean()


def hamming_loss_on_conf_matrix(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
    normalize: bool = True,
) -> Number:
    """
    Calculates Hamming loss from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    Hamming loss is the fraction of labels that are incorrectly predicted.
    If normalize is False, returns the mean of false positives and false negatives across all labels.
    """
    return binary_0_1_loss_on_conf_matrix(tp, fp, fn, tn, normalize=normalize).mean()


binary_accuracy = make_metric_on_y_true_and_y_pred(
    binary_accuracy_on_conf_matrix, "accuracy"
)
binary_0_1_loss = make_metric_on_y_true_and_y_pred(
    binary_0_1_loss_on_conf_matrix, "0/1 loss"
)
hamming_score = make_metric_on_y_true_and_y_pred(
    hamming_score_on_conf_matrix, "Hamming score"
)
hamming_loss = make_metric_on_y_true_and_y_pred(
    hamming_loss_on_conf_matrix, "Hamming loss"
)


########################################################################################
# Precision at k
########################################################################################


def binary_precision_at_k_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix, None],
    fn: Union[Number, DenseMatrix, None],
    tn: Union[Number, DenseMatrix, None],
    k: int,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates binary precision at k based on the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    .. math::
        \text{precision@k} = \frac{TP}{k}

    where :math:`k` is the number of positive labels per instance.
    """
    return tp / k


def precision_at_k_on_conf_matrix(
    tp: np.ndarray,
    fp: Union[np.ndarray, None],
    fn: Union[np.ndarray, None],
    tn: Union[np.ndarray, None],
    k: int,
) -> Number:
    """
    Calculates precision at k from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.
    """
    return binary_precision_at_k_on_conf_matrix(tp, fp, fn, tn, k).sum()


precision_at_k = make_metric_on_y_true_and_y_pred(
    precision_at_k_on_conf_matrix, "precision at k", skip_tn=True
)


########################################################################################
# Precision
########################################################################################


def binary_precision_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix, None],
    tn: Union[Number, DenseMatrix, None],
    epsilon: float = 1e-6,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates binary precision
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    Precision is the fraction of predicted positives that are actually positive.

    .. math::
        \text{precision} = \frac{TP}{TP + FP + \epsilon}

    where :math:`\epsilon` is a very small number to avoid division by zero.
    """
    return tp / (tp + fp + epsilon)


macro_precision_on_conf_matrix = make_macro_metric_on_conf_matrix(
    binary_precision_on_conf_matrix, "precision"
)
micro_precision_on_conf_matrix = make_micro_metric_on_conf_matrix(
    binary_precision_on_conf_matrix, "precision"
)
binary_precision = make_metric_on_y_true_and_y_pred(
    binary_precision_on_conf_matrix, "binary precision", skip_tn=True
)
macro_precision = make_metric_on_y_true_and_y_pred(
    macro_precision_on_conf_matrix, "macro-averaged precision", skip_tn=True
)
micro_precision = make_metric_on_y_true_and_y_pred(
    micro_precision_on_conf_matrix, "micro-averaged precision", skip_tn=True
)


########################################################################################
# Recall
########################################################################################


def binary_recall_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix, None],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix, None],
    epsilon: float = 1e-6,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates binary recall
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    Recall is the fraction of positives that were correctly predicted.

    .. math::
        \text{recall} = \frac{TP}{TP + FN + \epsilon}

    where :math:`\epsilon` is a very small number to avoid division by zero.
    """
    return tp / (tp + fn + epsilon)


macro_recall_on_conf_matrix = make_macro_metric_on_conf_matrix(
    binary_recall_on_conf_matrix, "recall"
)
micro_recall_on_conf_matrix = make_micro_metric_on_conf_matrix(
    binary_recall_on_conf_matrix, "recall"
)
binary_recall = make_metric_on_y_true_and_y_pred(
    binary_recall_on_conf_matrix, "binary recall", skip_tn=True
)
macro_recall = make_metric_on_y_true_and_y_pred(
    macro_recall_on_conf_matrix, "macro-averaged recall", skip_tn=True
)
micro_recall = make_metric_on_y_true_and_y_pred(
    micro_recall_on_conf_matrix, "micro-averaged recall", skip_tn=True
)


########################################################################################
# F-score
########################################################################################


def binary_fbeta_score_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix, None],
    beta: float = 1.0,
    epsilon: float = 1e-6,
) -> Union[Number, DenseMatrix]:
    r"""
    Compute the binary F-beta score.
    from the given true positives, false positives, false negatives and true negatives.

    The F-beta score is the weighted harmonic mean of precision and recall.

    .. math::
        F_\beta = (1 + \beta^2) \cdot \frac{TP}{\beta^2 \cdot (TP + FP) + TP + FN + \epsilon}

    where :math:`\epsilon` is a very small number to avoid division by zero.
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
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix, None],
    epsilon: float = 1e-6,
) -> Union[Number, DenseMatrix]:
    """
    Calculates binary F1 score, also known as balanced F-score or F-measure
    from the given true positives, false positives, false negatives and true negatives.
    This is an alias for :func:`binary_fbeta_score_on_conf_matrix` with beta=1.0.
    """
    return binary_fbeta_score_on_conf_matrix(tp, fp, fn, tn, beta=1.0, epsilon=epsilon)


macro_fbeta_score_on_conf_matrix = make_macro_metric_on_conf_matrix(
    binary_fbeta_score_on_conf_matrix, "F-beta score"
)
micro_fbeta_score_on_conf_matrix = make_micro_metric_on_conf_matrix(
    binary_fbeta_score_on_conf_matrix, "F-beta score"
)
binary_fbeta_score = make_metric_on_y_true_and_y_pred(
    binary_fbeta_score_on_conf_matrix, "binary F-beta score", skip_tn=True
)
macro_fbeta_score = make_metric_on_y_true_and_y_pred(
    macro_fbeta_score_on_conf_matrix, "macro-averaged F-beta score", skip_tn=True
)
micro_fbeta_score = make_metric_on_y_true_and_y_pred(
    micro_fbeta_score_on_conf_matrix, "micro-averaged F-beta score", skip_tn=True
)

macro_f1_score_on_conf_matrix = make_macro_metric_on_conf_matrix(
    binary_fbeta_score_on_conf_matrix, "F1 score"
)
micro_f1_score_on_conf_matrix = make_micro_metric_on_conf_matrix(
    binary_fbeta_score_on_conf_matrix, "F1 score"
)
binary_f1_score = make_metric_on_y_true_and_y_pred(
    binary_fbeta_score_on_conf_matrix, "binary F1 score"
)
macro_f1_score = make_metric_on_y_true_and_y_pred(
    macro_fbeta_score_on_conf_matrix, "macro-averaged F1 score"
)
micro_f1_score = make_metric_on_y_true_and_y_pred(
    micro_fbeta_score_on_conf_matrix, "micro-averaged F1 score"
)


########################################################################################
# Jaccard score
########################################################################################


def binary_jaccard_score_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix, None],
    epsilon: float = 1e-6,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates Jaccard score
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    Jaccard score is the intersection over union of predicted and true labels.

    .. math::
        J = \frac{TP}{TP + FP + FN + \epsilon}

    where :math:`\epsilon` is a very small number to avoid division by zero.
    """
    return tp / (tp + fp + fn + epsilon)


macro_jaccard_score_on_conf_matrix = make_macro_metric_on_conf_matrix(
    binary_jaccard_score_on_conf_matrix, "Jaccard score"
)
micro_jaccard_score_on_conf_matrix = make_micro_metric_on_conf_matrix(
    binary_jaccard_score_on_conf_matrix, "Jaccard score"
)
binary_jaccard_score = make_metric_on_y_true_and_y_pred(
    binary_jaccard_score_on_conf_matrix, "binary Jaccard score", skip_tn=True
)
macro_jaccard_score = make_metric_on_y_true_and_y_pred(
    macro_jaccard_score_on_conf_matrix, "macro-averaged Jaccard score", skip_tn=True
)
micro_jaccard_score = make_metric_on_y_true_and_y_pred(
    micro_jaccard_score_on_conf_matrix, "micro-averaged Jaccard score", skip_tn=True
)


########################################################################################
# Balanced accuracy
########################################################################################


def binary_balanced_accuracy_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix],
    epsilon: float = 1e-6,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates ballanced accuracy
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    Balanced accuracy is the average of true positive rate and true negative rate.

    .. math::
        \text{balanced accuracy} = \frac{TP}{TP + FN + \epsilon} + \frac{TN}{TN + FP + \epsilon}

    where :math:`\epsilon` is a very small number to avoid division by zero.
    """
    tpr = tp / (tp + fn + epsilon)
    tnr = tn / (tn + fp + epsilon)
    return (tpr + tnr) / 2


macro_balanced_accuracy_on_conf_matrix = make_macro_metric_on_conf_matrix(
    binary_balanced_accuracy_on_conf_matrix, "balanced accuracy"
)
micro_balanced_accuracy_on_conf_matrix = make_micro_metric_on_conf_matrix(
    binary_balanced_accuracy_on_conf_matrix, "balanced accuracy"
)
binary_balanced_accuracy = make_metric_on_y_true_and_y_pred(
    binary_balanced_accuracy_on_conf_matrix, "binary balanced accuracy"
)
macro_balanced_accuracy = make_metric_on_y_true_and_y_pred(
    macro_balanced_accuracy_on_conf_matrix, "macro-averaged balanced accuracy"
)
micro_balanced_accuracy = make_metric_on_y_true_and_y_pred(
    micro_balanced_accuracy_on_conf_matrix, "micro-averaged balanced accuracy"
)


########################################################################################
# G-mean
########################################################################################


def binary_gmean_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix],
    epsilon: float = 1e-6,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates G-mean (geometric mean)
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    G-mean is the square root of the product of true positive rate and true negative rate.

    .. math::
        \text{G-mean} = \sqrt{\frac{TP}{TP + FN + \epsilon} \times \text{TNR} = \frac{TN}{TN + FP + \epsilon}}

    where :math:`\epsilon` is a very small number to avoid division by zero.
    """
    tpr = tp / (tp + fn + epsilon)
    tnr = tn / (tn + fp + epsilon)
    return (tpr * tnr) ** 0.5


macro_gmean_on_conf_matrix = make_macro_metric_on_conf_matrix(
    binary_gmean_on_conf_matrix, "G-mean"
)
micro_gmean_on_conf_matrix = make_micro_metric_on_conf_matrix(
    binary_gmean_on_conf_matrix, "G-mean"
)
binary_gmean = make_metric_on_y_true_and_y_pred(
    binary_gmean_on_conf_matrix, "binary G-mean"
)
macro_gmean = make_metric_on_y_true_and_y_pred(
    macro_gmean_on_conf_matrix, "macro-averaged G-mean"
)
micro_gmean = make_metric_on_y_true_and_y_pred(
    micro_gmean_on_conf_matrix, "micro-averaged G-mean"
)


########################################################################################
# H-mean
########################################################################################


def binary_hmean_on_conf_matrix(
    tp: Union[Number, DenseMatrix],
    fp: Union[Number, DenseMatrix],
    fn: Union[Number, DenseMatrix],
    tn: Union[Number, DenseMatrix],
    epsilon: float = 1e-6,
) -> Union[Number, DenseMatrix]:
    r"""
    Calculates H-mean (harmonic mean)
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    H-mean is the harmonic mean of true positive rate and true negative rate.

    .. math::
        \text{H-mean} = \frac{2 \times \text{TPR} \times \text{TNR}}{\text{TPR} + \text{TNR}}

    where :math:`\text{TPR} = \frac{TP}{TP + FN + \epsilon}` and :math:`\text{TNR} = \frac{TN}{TN + FP + \epsilon}`
    and :math:`\epsilon` is a very small number to avoid division by zero.
    """
    tpr = tp / (tp + fn + epsilon)
    tnr = tn / (tn + fp + epsilon)
    return (2 * tpr * tnr) / (tpr + tnr)


macro_hmean_on_conf_matrix = make_macro_metric_on_conf_matrix(
    binary_hmean_on_conf_matrix, "H-mean"
)
micro_hmean_on_conf_matrix = make_micro_metric_on_conf_matrix(
    binary_hmean_on_conf_matrix, "H-mean"
)
binary_hmean = make_metric_on_y_true_and_y_pred(
    binary_hmean_on_conf_matrix, "binary H-mean"
)
macro_hmean = make_metric_on_y_true_and_y_pred(
    macro_hmean_on_conf_matrix, "macro-averaged H-mean"
)
micro_hmean = make_metric_on_y_true_and_y_pred(
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
    r"""
    Calculates coverage
    from the given entries of confusion matrix:
    true positives, false positives, false negatives, and true negatives.

    Coverage is the fraction of instances that have at least one positive label.

    .. math::
        \text{coverage} = \sum_{i=1}^m \frac{TP_i > 0}{m}

    where :math:`m` is the number of labels.
    """
    return (tp > 0).mean()


coverage = make_metric_on_y_true_and_y_pred(
    coverage_on_conf_matrix, "coverage", skip_tn=True
)
