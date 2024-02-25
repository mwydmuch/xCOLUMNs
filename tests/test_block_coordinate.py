import numpy as np
import torch
from scipy.sparse import csr_matrix

from xcolumns.block_coordinate import predict_using_bc_with_0approx
from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.metrics import bin_fmeasure_on_conf_matrix, macro_fmeasure_on_conf_matrix
from xcolumns.weighted_prediction import predict_top_k


def _run_block_coordinate(y_test, y_proba_test, k, init_pred):
    y_pred, meta = predict_using_bc_with_0approx(
        y_proba_test,
        bin_fmeasure_on_conf_matrix,
        k,
        return_meta=True,
        seed=2024,
        init_pred=init_pred,
    )

    assert type(y_pred) == type(y_proba_test)
    assert (y_pred.sum(axis=1) == k).all()
    return calculate_confusion_matrix(y_test, y_pred, normalize=False, skip_tn=False)


def test_frank_wolfe(generated_test_data):
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

    # Run numpy implementation
    np_C = _run_block_coordinate(y_test, y_proba_test, k, top_k_y_pred)

    # Run csr_matrix implementation
    csr_C = _run_frank_wolfe(
        csr_matrix(y_test), csr_matrix(y_proba_test), k, csr_matrix(top_k_y_pred)
    )

    # Run torch implementation
    torch_C = _run_frank_wolfe(
        torch.tensor(y_test), torch.tensor(y_proba_test), k, torch.tensor(top_k_y_pred)
    )

    # Convert torch tensors to numpy arrays for easier comparison
    torch_C.tp = torch_C.tp.numpy()
    torch_C.fp = torch_C.fp.numpy()
    torch_C.fn = torch_C.fn.numpy()
    torch_C.tn = torch_C.tn.numpy()

    assert np_C == csr_C == torch_C

    # Compare with top_k
    assert macro_fmeasure_on_conf_matrix(
        np_C.tp, np_C.fp, np_C.fn, np_C.tn
    ) > macro_fmeasure_on_conf_matrix(top_k_C.tp, top_k_C.fp, top_k_C.fn, top_k_C.tn)
