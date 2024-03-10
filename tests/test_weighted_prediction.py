import numpy as np
from pytest import _report_data_type, _test_prediction_method_with_different_types
from scipy.sparse import csr_matrix

from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.utils import *
from xcolumns.weighted_prediction import (
    predict_for_optimal_macro_balanced_accuracy,
    predict_weighted_per_instance,
)


def _run_and_test_weighted_prediction(y_true, y_proba, k, a, b):
    _report_data_type(y_proba)
    y_pred, meta = predict_weighted_per_instance(y_proba, k, a=a, b=b, return_meta=True)
    print(f"  time={meta['time']}s")

    assert type(y_pred) == type(y_proba)
    assert y_pred.dtype == y_proba.dtype
    assert (y_pred.sum(axis=1) == k).all()

    return (
        calculate_confusion_matrix(y_true, y_pred, normalize=False, skip_tn=False),
        y_pred,
    )


def test_weighted_prediction(generated_test_data):
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

    _test_prediction_method_with_different_types(
        _run_and_test_weighted_prediction,
        (y_test, y_proba_test, k, a, b),
    )


def _run_and_test_prediction_for_optimal_macro_balanced_accuracy(
    y_true, y_proba, k, priors
):
    _report_data_type(y_proba)
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


def test_prediction_for_optimal_macro_balanced_accuracy(generated_test_data):
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

    _test_prediction_method_with_different_types(
        _run_and_test_prediction_for_optimal_macro_balanced_accuracy,
        (y_test, y_proba_test, k, priors),
    )
