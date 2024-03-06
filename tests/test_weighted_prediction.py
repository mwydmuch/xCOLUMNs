import numpy as np
from scipy.sparse import csr_matrix

from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.utils import *
from xcolumns.weighted_prediction import (
    predict_for_optimal_macro_balanced_accuracy,
    predict_weighted_per_instance,
)


def _run_and_test_weighted_prediction(y_true, y_proba, k, a, b):
    print(f"input dtype={y_proba.dtype}")
    if isinstance(y_proba, csr_matrix):
        print(
            f"  csr_matrix nnz={y_proba.nnz}, shape={y_proba.shape}, sparsity={y_proba.nnz / y_proba.shape[0] / y_proba.shape[1]}"
        )
    y_pred, meta = predict_weighted_per_instance(y_proba, k, a=a, b=b, return_meta=True)
    print(f"  time={meta['time']}s")

    assert type(y_pred) == type(y_proba)
    assert y_pred.dtype == y_proba.dtype
    assert (y_pred.sum(axis=1) == k).all()

    return (
        calculate_confusion_matrix(y_true, y_pred, normalize=False, skip_tn=False),
        y_pred,
    )


def test_weighted_prediction(generated_test_data, test_method):
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        y_proba_train,
        y_proba_val,
        y_proba_test,
    ) = generated_test_data
    k = 3

    # Generate random weights
    a = np.random.rand(y_proba_train.shape[1])
    b = np.random.rand(y_proba_train.shape[1])
    # b = np.zeros(y_proba_train.shape[1])

    test_method(
        _run_and_test_weighted_prediction,
        (y_test, y_proba_test, k, a, b),
    )


def _run_and_test_prediction_for_optimal_macro_balanced_accuracy(
    y_true, y_proba, k, priors
):
    print(f"input dtype={y_proba.dtype}")
    if isinstance(y_proba, csr_matrix):
        print(
            f"  csr_matrix nnz={y_proba.nnz}, shape={y_proba.shape}, sparsity={y_proba.nnz / y_proba.shape[0] / y_proba.shape[1]}"
        )
    y_pred, meta = predict_for_optimal_macro_balanced_accuracy(
        y_proba, k, priors, return_meta=True
    )
    print(f"  time={meta['time']}s")

    assert type(y_pred) == type(y_proba)
    assert y_pred.dtype == y_proba.dtype
    assert (y_pred.sum(axis=1) == k).all()

    return (
        calculate_confusion_matrix(y_true, y_pred, normalize=False, skip_tn=False),
        y_pred,
    )


def test_prediction_for_optimal_macro_balanced_accuracy(
    generated_test_data, test_method
):
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        y_proba_train,
        y_proba_val,
        y_proba_test,
    ) = generated_test_data
    k = 3

    # Calculate priors
    priors = y_train.mean(axis=0)

    test_method(
        _run_and_test_prediction_for_optimal_macro_balanced_accuracy,
        (y_test, y_proba_test, k, priors),
    )
