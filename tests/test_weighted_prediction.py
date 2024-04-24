import numpy as np
from pytest import _report_data_type, _test_prediction_method_with_different_types
from scipy.sparse import csr_matrix

from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.metrics import jpv_inverse_propensities, label_priors
from xcolumns.utils import *
from xcolumns.weighted_prediction import (
    predict_log_weighted_per_instance,
    predict_optimizing_instance_precision,
    predict_optimizing_instance_propensity_scored_precision,
    predict_optimizing_macro_balanced_accuracy,
    predict_optimizing_macro_recall,
    predict_power_law_weighted_per_instance,
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
    y_test = generated_test_data["y_test"]
    y_proba_test = generated_test_data["y_proba_test"]
    k = 3

    # Generate random weights
    a = np.random.rand(y_proba_test.shape[1])
    b = np.random.rand(y_proba_test.shape[1])
    # b = np.zeros(y_proba_train.shape[1])

    _test_prediction_method_with_different_types(
        _run_and_test_weighted_prediction,
        (y_test, y_proba_test, k, a, b),
    )


def _run_and_test_prediction_optimizing_macro_balanced_accuracy(
    y_true, y_proba, k, priors
):
    _report_data_type(y_proba)
    y_pred, meta = predict_optimizing_macro_balanced_accuracy(
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


def test_prediction_optizming_macro_balanced_accuracy(generated_test_data):
    y_train = generated_test_data["y_train"]
    y_test = generated_test_data["y_test"]
    y_proba_test = generated_test_data["y_proba_test"]
    k = 3

    # Calculate priors
    priors = y_train.mean(axis=0)

    _test_prediction_method_with_different_types(
        _run_and_test_prediction_optimizing_macro_balanced_accuracy,
        (y_test, y_proba_test, k, priors),
    )


def test_wrapper_methods(generated_test_data):
    y_test = generated_test_data["y_test"]
    y_proba_test = generated_test_data["y_proba_test"]

    priors = label_priors(y_test)
    inverse_propensities = jpv_inverse_propensities(y_test)

    y_pred = predict_optimizing_instance_precision(y_proba_test, k=3)
    y_pred = predict_log_weighted_per_instance(y_proba_test, k=3, priors=priors)
    y_pred = predict_power_law_weighted_per_instance(
        y_proba_test, k=3, priors=priors, beta=0.5
    )
    y_pred = predict_optimizing_instance_propensity_scored_precision(
        y_proba_test, k=3, inverse_propensities=inverse_propensities
    )
    y_pred = predict_optimizing_macro_recall(y_proba_test, k=3, priors=priors)
