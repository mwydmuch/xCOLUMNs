import numpy as np
import torch
from scipy.sparse import csr_matrix

from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.frank_wolfe import find_optimal_randomized_classifier_using_frank_wolfe
from xcolumns.metrics import macro_fmeasure_on_conf_matrix
from xcolumns.weighted_prediction import predict_top_k


def _run_frank_wolfe(y_val, y_proba_val, y_test, y_proba_test, k, init_a, init_b):
    print(f"input dtype={y_proba_val.dtype}")
    if isinstance(y_proba_val, csr_matrix):
        print(
            f"  csr_matrix nnz={y_proba_val.nnz}, shape={y_proba_val.shape}, sparsity={y_proba_val.nnz / y_proba_val.shape[0] / y_proba_val.shape[1]}"
        )
    rnd_clf, meta = find_optimal_randomized_classifier_using_frank_wolfe(
        y_val,
        y_proba_val,
        macro_fmeasure_on_conf_matrix,
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


def test_frank_wolfe(generated_test_data, test_method):
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

    # Generate initial random classifier
    init_a = np.random.rand(y_proba_train.shape[1])
    init_b = np.random.rand(y_proba_train.shape[1])

    # Run predict_top_k to get baseline classifier
    top_k_y_pred = predict_top_k(y_proba_test, k)
    top_k_C = calculate_confusion_matrix(
        y_test, top_k_y_pred, normalize=False, skip_tn=False
    )

    conf_mats, y_preds = test_method(
        _run_frank_wolfe,
        (y_val, y_proba_val, y_test, y_proba_test, k, init_a, init_b),
    )

    # Compare with top_k
    fw_score = macro_fmeasure_on_conf_matrix(
        conf_mats[0].tp, conf_mats[0].fp, conf_mats[0].fn, conf_mats[0].tn
    )
    top_k_score = macro_fmeasure_on_conf_matrix(
        top_k_C.tp, top_k_C.fp, top_k_C.fn, top_k_C.tn
    )
    print(f"fw_score={fw_score}, top_k_score={top_k_score}")
    assert fw_score >= top_k_score
