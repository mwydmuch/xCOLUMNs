import numpy as np
import torch
from scipy.sparse import csr_matrix

from xcolumns.block_coordinate import predict_using_bc_with_0approx
from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.metrics import bin_fmeasure_on_conf_matrix, macro_fmeasure_on_conf_matrix
from xcolumns.weighted_prediction import predict_top_k


def _run_block_coordinate(y_test, y_proba_test, k, init_pred):
    print(f"input dtype={y_proba_test.dtype}")
    if isinstance(y_proba_test, csr_matrix):
        print(
            f"  csr_matrix nnz={y_proba_test.nnz}, shape={y_proba_test.shape}, sparsity={y_proba_test.nnz / y_proba_test.shape[0] / y_proba_test.shape[1]}"
        )
    y_pred, meta = predict_using_bc_with_0approx(
        y_proba_test,
        bin_fmeasure_on_conf_matrix,
        k,
        return_meta=True,
        seed=2024,
        init_pred=init_pred,
    )

    assert type(y_pred) == type(y_proba_test)
    assert y_pred.dtype == y_proba_test.dtype
    assert (y_pred.sum(axis=1) == k).all()
    return (
        calculate_confusion_matrix(y_test, y_pred, normalize=False, skip_tn=False),
        y_pred,
    )


def test_block_coordinate(generated_test_data, test_method):
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

    conf_mats, y_preds = test_method(
        _run_block_coordinate,
        (y_test, y_proba_test, k, top_k_y_pred),
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
