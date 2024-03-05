from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

from .numba_csr_functions import *
from .types import *
from .utils import *


########################################################################################
# Confusion matrix class
########################################################################################


class ConfusionMatrix:
    """
    Class representing a confusion matrix.

    When unpacked returns (tp, fp, fn, tn) (in this order).
    """

    def __init__(
        self,
        tp: Union[Number, DenseMatrix],
        fp: Union[Number, DenseMatrix],
        fn: Union[Number, DenseMatrix],
        tn: Union[Number, DenseMatrix],
    ):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    # For compatibility with methods that expect a tuple
    def __iter__(self):
        yield self.tp
        yield self.fp
        yield self.fn
        yield self.tn

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfusionMatrix):
            return False

        if isinstance(self.tp, Number):
            return (
                self.tp == other.tp
                and self.fp == other.fp
                and self.fn == other.fn
                and self.tn == other.tn
            )
        elif isinstance(self.tp, DenseMatrix):
            return (
                (self.tp == other.tp).all()
                and (self.fp == other.fp).all()
                and (self.fn == other.fn).all()
                and (self.tn == other.tn).all()
            )

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        return ConfusionMatrix(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tn + other.tn,
        )

    def __iadd__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        return self

    def __sub__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        return ConfusionMatrix(
            self.tp - other.tp,
            self.fp - other.fp,
            self.fn - other.fn,
            self.tn - other.tn,
        )

    def __isub__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        self.tp -= other.tp
        self.fp -= other.fp
        self.fn -= other.fn
        self.tn -= other.tn
        return self

    def __mul__(self, other: Union[Number, DenseMatrix]) -> "ConfusionMatrix":
        return ConfusionMatrix(
            self.tp * other,
            self.fp * other,
            self.fn * other,
            self.tn * other,
        )

    def __imul__(self, other: Union[Number, DenseMatrix]) -> "ConfusionMatrix":
        self.tp *= other
        self.fp *= other
        self.fn *= other
        self.tn *= other
        return self

    def __truediv__(self, other: Union[Number, DenseMatrix]) -> "ConfusionMatrix":
        return ConfusionMatrix(
            self.tp / other,
            self.fp / other,
            self.fn / other,
            self.tn / other,
        )

    def __itruediv__(self, other: Union[Number, DenseMatrix]) -> "ConfusionMatrix":
        self.tp /= other
        self.fp /= other
        self.fn /= other
        self.tn /= other
        return self

    def __floordiv__(self, other: Union[Number, DenseMatrix]) -> "ConfusionMatrix":
        return ConfusionMatrix(
            self.tp // other,
            self.fp // other,
            self.fn // other,
            self.tn // other,
        )

    def __ifloordiv__(self, other: Union[Number, DenseMatrix]) -> "ConfusionMatrix":
        self.tp //= other
        self.fp //= other
        self.fn //= other
        self.tn //= other
        return self


########################################################################################
# Functions to calculate/update confusion matrix
########################################################################################


def _calculate_tp_dense(y_true: DenseMatrix, y_pred: DenseMatrix) -> DenseMatrix:
    return (y_true * y_pred).sum(axis=0)


# Alternative version, performance is similar
# def _calculate_tp_csr(y_true: csr_matrix, y_pred: csr_matrix):
#     return (y_pred.multiply(y_true)).sum(axis=0)


def _calculate_tp_csr(y_true: csr_matrix, y_pred: csr_matrix) -> np.ndarray:
    n, m = y_true.shape
    return numba_calculate_sum_0_csr_mat_mul_mat(
        *unpack_csr_matrices(y_pred, y_true), n, m
    )


def _calculate_fp_dense(y_true: DenseMatrix, y_pred: DenseMatrix) -> DenseMatrix:
    return ((1 - y_true) * y_pred).sum(axis=0)


def _calculate_fp_csr_slow(y_true: csr_matrix, y_pred: csr_matrix) -> np.ndarray:
    n, m = y_true.shape
    fp = np.zeros(m, dtype=FLOAT_TYPE)
    dense_ones = np.ones(m, dtype=FLOAT_TYPE)
    for i in range(n):
        fp += y_pred[i].multiply(dense_ones - y_true[i])
    return fp


def _calculate_fn_dense(y_true: DenseMatrix, y_pred: DenseMatrix) -> DenseMatrix:
    return (y_true * (1 - y_pred)).sum(axis=0)


def _calculate_fp_csr(y_true: csr_matrix, y_pred: csr_matrix) -> np.ndarray:
    n, m = y_true.shape
    return numba_calculate_sum_0_csr_mat_mul_ones_minus_mat(
        *unpack_csr_matrices(y_pred, y_true), n, m
    )


def _calculate_fn_csr_slow(y_true: csr_matrix, y_pred: csr_matrix) -> np.ndarray:
    n, m = y_true.shape
    fn = np.zeros(m, dtype=FLOAT_TYPE)
    dense_ones = np.ones(m, dtype=FLOAT_TYPE)
    for i in range(n):
        fn += y_true[i].multiply(dense_ones - y_pred[i])

    return fn


def _calculate_fn_csr(y_true: csr_matrix, y_pred: csr_matrix) -> np.ndarray:
    n, m = y_true.shape
    return numba_calculate_sum_0_csr_mat_mul_ones_minus_mat(
        *unpack_csr_matrices(y_true, y_pred), n, m
    )


def _calculate_tn_dense(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (1 - y_true) * (1 - y_pred).sum(axis=0)


def _calculate_conf_mat_entry(
    y_true: Matrix,
    y_pred: Matrix,
    func_for_dense: Callable,
    func_for_csr: Callable,
    normalize: bool = False,
) -> DenseMatrix:
    if isinstance(y_true, DenseMatrix) and isinstance(y_pred, DenseMatrix):
        func = func_for_dense
    elif isinstance(y_true, csr_matrix) and isinstance(y_pred, csr_matrix):
        func = func_for_csr
    else:
        raise ValueError(
            "y_true and y_pred must be both np.ndarray, both torch.Tensor, or csr_matrix"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    val = func(y_true, y_pred)

    if normalize:
        val = val / y_true.shape[0]

    return val


def calculate_tp(
    y_true: Matrix,
    y_pred: Matrix,
    normalize: bool = False,
) -> DenseMatrix:
    """
    Calculate true positives for the given true and predicted labels.
    """
    return _calculate_conf_mat_entry(
        y_true, y_pred, _calculate_tp_dense, _calculate_tp_csr, normalize=normalize
    )


def calculate_fp(
    y_true: Matrix,
    y_pred: Matrix,
    normalize: bool = False,
) -> DenseMatrix:
    """
    Calculate false positives for the given true and predicted labels.
    """
    return _calculate_conf_mat_entry(
        y_true, y_pred, _calculate_fp_dense, _calculate_fp_csr, normalize=normalize
    )


def calculate_fn(
    y_true: Matrix,
    y_pred: Matrix,
    normalize: bool = False,
) -> DenseMatrix:
    """
    Calculate false negatives for the given true and predicted labels.
    """
    return _calculate_conf_mat_entry(
        y_true, y_pred, _calculate_fn_dense, _calculate_fn_csr, normalize=normalize
    )


def calculate_confusion_matrix(
    y_true: Matrix,
    y_pred: Matrix,
    normalize: bool = False,
    skip_tn: bool = False,
) -> ConfusionMatrix:
    """
    Calculate confusion matrix for given true and predicted labels.
    """
    tp = calculate_tp(y_true, y_pred, normalize=normalize)
    fp = calculate_fp(y_true, y_pred, normalize=normalize)
    fn = calculate_fn(y_true, y_pred, normalize=normalize)

    n, m = y_true.shape
    if skip_tn:
        tn = tp.copy()
        tn[:] = -1
    else:
        tn = -tp - fp - fn + (1.0 if normalize else n)

    return ConfusionMatrix(tp, fp, fn, tn)


def update_unnormalized_confusion_matrix(
    C: ConfusionMatrix,
    y_true: Matrix,
    y_pred: Matrix,
    skip_tn: bool = False,
) -> None:
    """
    Updates the given unnormalized confusion matrix in place based on provided true and predicted labels for a single instance.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if isinstance(y_true, DenseMatrix) and isinstance(y_pred, DenseMatrix):
        C.tp += y_true * y_pred
        C.fp += (1 - y_true) * y_pred
        C.fn += y_true * (1 - y_pred)
        if not skip_tn:
            C.tn += (1 - y_true) * (1 - y_pred)
    elif isinstance(y_true, csr_matrix) and isinstance(y_pred, csr_matrix):
        numba_add_to_unnormalized_confusion_matrix_csr(
            C.tp,
            C.fp,
            C.fn,
            C.tn,
            y_true.data,
            y_true.indices,
            y_pred.data,
            y_pred.indices,
            skip_tn=skip_tn,
        )
    else:
        raise ValueError(
            "y_true and y_pred must be both np.ndarray, both torch.Tensor, or both csr_matrix"
        )
