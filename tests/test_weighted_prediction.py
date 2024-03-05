import numpy as np
import torch
from scipy.sparse import csr_matrix

from xcolumns.confusion_matrix import calculate_confusion_matrix
from xcolumns.weighted_prediction import predict_weighted_per_instance


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
    # b = np.random.rand(y_proba_train.shape[1])
    b = np.zeros(y_proba_train.shape[1])

    # Run numpy implementation
    y_proba_test = y_proba_test.astype(np.float64)
    np_C_64, np_pred_64 = _run_and_test_weighted_prediction(
        y_test, y_proba_test, k, a, b
    )

    y_proba_test = y_proba_test.astype(np.float32)
    np_C_32, np_pred_32 = _run_and_test_weighted_prediction(
        y_test, y_proba_test, k, a, b
    )

    # Run csr_matrix implementation
    csr_C, csr_pred = _run_and_test_weighted_prediction(
        csr_matrix(y_test), csr_matrix(y_proba_test), k, a, b
    )

    # Run torch implementation
    torch_C, torch_pred = _run_and_test_weighted_prediction(
        torch.tensor(y_test),
        torch.tensor(y_proba_test),
        k,
        torch.tensor(a),
        torch.tensor(b),
    )

    # Convert torch tensors to numpy arrays for easier comparison
    torch_C.tp = torch_C.tp.numpy()
    torch_C.fp = torch_C.fp.numpy()
    torch_C.fn = torch_C.fn.numpy()
    torch_C.tn = torch_C.tn.numpy()

    # Compere if all implementations are equal
    assert np_C_64 == np_C_32 == csr_C == torch_C
