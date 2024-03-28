# Create data for testing using sklearn

import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier


try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture(autouse=True, scope="session")
def generated_test_data():
    """
    Generate data for testing.
    """
    seed = 2024
    max_top_k = 5
    x, y = make_multilabel_classification(
        n_samples=100000,
        n_features=50,
        n_classes=25,
        n_labels=3,
        length=25,
        allow_unlabeled=True,
        sparse=False,
        return_indicator="dense",
        return_distributions=False,
        random_state=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.3, random_state=seed
    )
    clf = MultiOutputClassifier(LogisticRegression()).fit(x_train, y_train)

    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert y_train.shape[1] == y_val.shape[1] == y_test.shape[1]

    def clf_predict(clf, x, sparsfy=False):
        """
        Process the output of a multioutput classifier to get the marginal probabilities of each label (class).
        """
        y_proba = clf.predict_proba(x)

        # Convert the output of MultiOutputClassifier that contains the marginal probabilities for both P(y=0) and P(y=1) to just a matrix of P(y=1)
        y_proba = np.array(y_proba)[:, :, 1].transpose()

        # Sparify the matrix
        if sparsfy:
            top_k_thr = -np.partition(-y_proba, max_top_k, axis=1)[:, max_top_k]
            top_k_thr = top_k_thr.reshape((-1,))
            top_k_thr[top_k_thr >= 0.1] = 0.1
            y_proba[y_proba < top_k_thr[:, None]] = 0
            assert ((y_proba > 0).sum(axis=1) >= max_top_k).all()

        return y_proba

    y_proba_train = clf_predict(clf, x_train)
    y_proba_val = clf_predict(clf, x_val)
    y_proba_test = clf_predict(clf, x_test)

    assert y_proba_train.shape == y_train.shape
    assert y_proba_val.shape == y_val.shape
    assert y_proba_test.shape == y_test.shape

    test_data = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_proba_train": y_proba_train,
        "y_proba_val": y_proba_val,
        "y_proba_test": y_proba_test,
    }

    return test_data


def _report_data_type(data):
    print(f"input dtype={data.dtype}")
    if isinstance(data, csr_matrix):
        print(
            f"  csr_matrix nnz={data.nnz}, shape={data.shape}, sparsity={data.nnz / data.shape[0] / data.shape[1]}"
        )
    elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        print(f"  device={data.device}")


def _comparison_of_C_matrices(C1, C2, max_diff=5):
    if max_diff > 0:
        assert np.abs(C1.tp - C2.tp).max() <= max_diff
        assert np.abs(C1.fp - C2.fp).max() <= max_diff
        assert np.abs(C1.fn - C2.fn).max() <= max_diff
        assert np.abs(C1.tn - C2.tn).max() <= max_diff
    else:
        assert np.allclose(C1.tp, C2.tp)
        assert np.allclose(C1.fp, C2.fp)
        assert np.allclose(C1.fn, C2.fn)
        assert np.allclose(C1.tn, C2.tn)


def _test_prediction_method_with_different_types(
    method_wrapper, args, test_torch=True, max_diff=3
):
    """
    Test a method with different input types and compare the results.
    """
    conf_mats = []
    y_preds = []

    # Run numpy implementation for float64 and float32
    np_args_64 = [
        arg.astype(np.float64) if isinstance(arg, np.ndarray) else arg for arg in args
    ]
    np_C_64, np_pred_64 = method_wrapper(*np_args_64)
    conf_mats.append(np_C_64)
    y_preds.append(np_pred_64)

    np_args_32 = [
        arg.astype(np.float32) if isinstance(arg, np.ndarray) else arg for arg in args
    ]
    np_C_32, np_pred_32 = method_wrapper(*np_args_32)
    conf_mats.append(np_C_32)
    y_preds.append(np_pred_32)

    _comparison_of_C_matrices(np_C_64, np_C_32, max_diff=max_diff)

    # Run csr_matrix implementation
    csr_args = [
        csr_matrix(arg) if isinstance(arg, np.ndarray) and len(arg.shape) > 1 else arg
        for arg in args
    ]
    csr_C, csr_pred = method_wrapper(*csr_args)
    conf_mats.append(csr_C)
    y_preds.append(csr_pred)

    _comparison_of_C_matrices(np_C_64, csr_C, max_diff=max_diff)

    # Run torch implementation
    if TORCH_AVAILABLE and test_torch:
        torch_args = [
            torch.tensor(arg, dtype=torch.float32)
            if isinstance(arg, np.ndarray)
            else arg
            for arg in args
        ]
        torch_C, torch_pred = method_wrapper(*torch_args)
        conf_mats.append(torch_C)
        y_preds.append(torch_pred)

        # Run torch implementation on cuda
        if torch.cuda.is_available():
            torch_cuda_args = [
                torch.tensor(arg, device="cuda", dtype=torch.float32)
                if isinstance(arg, np.ndarray)
                else arg
                for arg in args
            ]
            torch_cuda_C, torch_cuda_pred = method_wrapper(*torch_cuda_args)
            conf_mats.append(torch_cuda_C)
            y_preds.append(torch_cuda_pred)

            torch_cuda_C.tp = torch_cuda_C.tp.cpu()
            torch_cuda_C.fp = torch_cuda_C.fp.cpu()
            torch_cuda_C.fn = torch_cuda_C.fn.cpu()
            torch_cuda_C.tn = torch_cuda_C.tn.cpu()

            _comparison_of_C_matrices(torch_cuda_C, torch_C, max_diff=max_diff)

        # Convert torch tensors to numpy arrays for easier comparison
        torch_C.tp = torch_C.tp.numpy()
        torch_C.fp = torch_C.fp.numpy()
        torch_C.fn = torch_C.fn.numpy()
        torch_C.tn = torch_C.tn.numpy()

        _comparison_of_C_matrices(np_C_64, torch_C, max_diff=max_diff)

    return conf_mats, y_preds


sys.modules[
    "pytest"
]._test_prediction_method_with_different_types = (
    _test_prediction_method_with_different_types
)
sys.modules["pytest"]._report_data_type = _report_data_type
