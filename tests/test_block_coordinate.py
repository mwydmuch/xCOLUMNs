import numpy as np
import torch
from pytest import _report_data_type, _test_prediction_method_with_different_types
from scipy.sparse import csr_matrix

from xcolumns.block_coordinate import predict_using_bc_with_0approx
from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.metrics import binary_recall_on_conf_matrix, macro_recall_on_conf_matrix
from xcolumns.weighted_prediction import predict_optimizing_macro_recall, predict_top_k


def _run_block_coordinate(y_test, y_proba_test, k, init_y_pred):
    _report_data_type(y_proba_test)
    y_pred, meta = predict_using_bc_with_0approx(
        y_proba_test,
        binary_recall_on_conf_matrix,
        k,
        return_meta=True,
        seed=2024,
        init_y_pred=init_y_pred,
    )
    print(f"  time={meta['time']}s")

    assert type(y_pred) == type(y_proba_test)
    assert y_pred.dtype == y_proba_test.dtype
    if k > 0:
        assert (y_pred.sum(axis=1) == k).all()
    return (
        calculate_confusion_matrix(y_test, y_pred, normalize=False, skip_tn=False),
        y_pred,
    )


def test_block_coordinate_arguments(generated_test_data):
    y_proba_test = generated_test_data["y_proba_test"]

    for init_y_pred in ["random", "greedy", "top"]:
        y_pred, meta = predict_using_bc_with_0approx(
            y_proba_test,
            binary_recall_on_conf_matrix,
            3,
            return_meta=True,
            seed=2024,
            init_y_pred=init_y_pred,
        )

    for k in [0, 3]:
        y_pred, meta = predict_using_bc_with_0approx(
            y_proba_test,
            binary_recall_on_conf_matrix,
            k,
            return_meta=True,
            seed=2024,
            init_y_pred="random",
        )


def test_block_coordinate_with_different_types(generated_test_data):
    y_test = generated_test_data["y_test"]
    y_proba_test = generated_test_data["y_proba_test"]
    k = 3

    # Run predict_top_k to get baseline classifier and initial prediction
    top_k_y_pred = predict_top_k(y_proba_test, k)
    top_k_C = calculate_confusion_matrix(
        y_test, top_k_y_pred, normalize=False, skip_tn=False
    )

    conf_mats, y_preds = _test_prediction_method_with_different_types(
        _run_block_coordinate,
        (y_test, y_proba_test, k, top_k_y_pred),
        test_torch=False,
    )

    # Compare with closed formula for recall
    opt_recall_y_pred = predict_optimizing_macro_recall(
        y_proba_test,
        k,
        priors=y_test.mean(axis=0),
    )
    opt_recall_C = calculate_confusion_matrix(
        y_test, opt_recall_y_pred, normalize=False, skip_tn=False
    )

    # Compare with top-k
    top_k_score = macro_recall_on_conf_matrix(*top_k_C)
    bc_score = macro_recall_on_conf_matrix(*conf_mats[0])
    opt_recall_score = macro_recall_on_conf_matrix(*opt_recall_C)
    print(
        f"Top-k score={top_k_score}, BC score={bc_score}, opt recall score={opt_recall_score}"
    )
    assert bc_score >= top_k_score
    assert abs(opt_recall_score - bc_score) < 0.02
