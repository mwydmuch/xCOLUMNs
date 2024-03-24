import numpy as np
import torch
from pytest import _report_data_type, _test_prediction_method_with_different_types
from scipy.sparse import csr_matrix

from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.frank_wolfe import find_optimal_randomized_classifier_using_frank_wolfe
from xcolumns.metrics import macro_recall_on_conf_matrix
from xcolumns.weighted_prediction import predict_optimizing_macro_recall, predict_top_k


def _run_frank_wolfe(y_val, y_proba_val, y_test, y_proba_test, k, init_a, init_b):
    _report_data_type(y_proba_val)
    rnd_clf, meta = find_optimal_randomized_classifier_using_frank_wolfe(
        y_val,
        y_proba_val,
        macro_recall_on_conf_matrix,
        k,
        return_meta=True,
        seed=2024,
        init_classifier=(init_a, init_b),
        verbose=True,
    )
    print(f"  time={meta['time']}s")

    y_pred = rnd_clf.predict(y_proba_test, seed=2024)
    assert type(y_pred) == type(y_proba_test)
    assert y_pred.dtype == y_proba_test.dtype
    assert (y_pred.sum(axis=1) == k).all()
    return (
        calculate_confusion_matrix(y_test, y_pred, normalize=False, skip_tn=False),
        y_pred,
    )


def __test_frank_wolfe_arguments(generated_test_data):
    y_val = generated_test_data["y_val"]
    y_proba_val = generated_test_data["y_proba_val"]

    for init_classifier in ["random", "top"]:
        rnd_classifier, meta = find_optimal_randomized_classifier_using_frank_wolfe(
            y_val,
            y_proba_val,
            macro_recall_on_conf_matrix,
            3,
            return_meta=True,
            seed=2024,
            init_classifier=init_classifier,
        )

    for k in [0, 3]:
        rnd_classifier, meta = find_optimal_randomized_classifier_using_frank_wolfe(
            y_val,
            y_proba_val,
            macro_recall_on_conf_matrix,
            k,
            return_meta=True,
            seed=2024,
            init_classifier=init_classifier,
        )


def test_frank_wolfe(generated_test_data):
    y_val = generated_test_data["y_val"]
    y_proba_val = generated_test_data["y_proba_val"]
    y_test = generated_test_data["y_test"]
    y_proba_test = generated_test_data["y_proba_test"]
    k = 3

    # Generate initial random classifier
    init_a = np.random.rand(y_proba_val.shape[1])
    init_b = np.random.rand(y_proba_val.shape[1])

    # Run predict_top_k to get baseline classifier
    top_k_y_pred = predict_top_k(y_proba_test, k)
    top_k_C = calculate_confusion_matrix(
        y_test, top_k_y_pred, normalize=False, skip_tn=False
    )

    conf_mats, y_preds = _test_prediction_method_with_different_types(
        _run_frank_wolfe,
        (y_val, y_proba_val, y_test, y_proba_test, k, init_a, init_b),
        test_torch=False,
    )

    # Compare with closed formula for recall
    opt_recall_y_pred = predict_optimizing_macro_recall(
        y_proba_test,
        k,
        priors=y_val.mean(axis=0),
    )
    opt_recall_C = calculate_confusion_matrix(
        y_test, opt_recall_y_pred, normalize=False, skip_tn=False
    )

    top_k_score = macro_recall_on_conf_matrix(*top_k_C)
    fw_score = macro_recall_on_conf_matrix(*conf_mats[0])
    opt_recall_score = macro_recall_on_conf_matrix(*opt_recall_C)
    print(
        f"top-k score={top_k_score}, FW score={fw_score}, opt recall score={opt_recall_score}"
    )
    assert fw_score >= top_k_score
    assert abs(opt_recall_score - fw_score) < 0.02
