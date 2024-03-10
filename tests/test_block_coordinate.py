import numpy as np
import torch
from pytest import _report_data_type, _test_prediction_method_with_different_types
from scipy.sparse import csr_matrix

from xcolumns.block_coordinate import predict_using_bc_with_0approx
from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.metrics import bin_fmeasure_on_conf_matrix, macro_fmeasure_on_conf_matrix
from xcolumns.weighted_prediction import predict_top_k


def _run_block_coordinate(y_test, y_proba_test, k, init_y_pred):
    _report_data_type(y_proba_test)
    y_pred, meta = predict_using_bc_with_0approx(
        y_proba_test,
        bin_fmeasure_on_conf_matrix,
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

    for init_y_pred in ["random", "greedy", "topk"]:
        y_pred, meta = predict_using_bc_with_0approx(
            y_proba_test,
            bin_fmeasure_on_conf_matrix,
            3,
            return_meta=True,
            seed=2024,
            init_y_pred=init_y_pred,
        )

    for k in [0, 3]:
        y_pred, meta = predict_using_bc_with_0approx(
            y_proba_test,
            bin_fmeasure_on_conf_matrix,
            k,
            return_meta=True,
            seed=2024,
            init_y_pred="random",
        )


def test_block_coordinate_with_different_types(generated_test_data):
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

    # Compare with top_k
    bc_score = macro_fmeasure_on_conf_matrix(
        conf_mats[0].tp, conf_mats[0].fp, conf_mats[0].fn, conf_mats[0].tn
    )
    top_k_score = macro_fmeasure_on_conf_matrix(
        top_k_C.tp, top_k_C.fp, top_k_C.fn, top_k_C.tn
    )
    print(f"bc_score={bc_score}, top_k_score={top_k_score}")
    assert bc_score >= top_k_score
