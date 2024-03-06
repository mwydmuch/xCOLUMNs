# Create data for testing using sklearn

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
        n_features=100,
        n_classes=50,
        n_labels=3,
        length=50,
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

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        y_proba_train,
        y_proba_val,
        y_proba_test,
    )


if TORCH_AVAILABLE:
    import torch


def _test_method(method_wrapper, args):
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

    assert np_C_64 == np_C_32

    # Run csr_matrix implementation
    csr_args = [
        csr_matrix(arg) if isinstance(arg, np.ndarray) and len(arg.shape) > 1 else arg
        for arg in args
    ]
    csr_C, csr_pred = method_wrapper(*csr_args)
    conf_mats.append(csr_C)
    y_preds.append(csr_pred)

    assert np_C_64 == csr_C

    # Run torch implementation
    if TORCH_AVAILABLE:
        torch_args = [
            torch.tensor(arg) if isinstance(arg, np.ndarray) else arg for arg in args
        ]
        torch_C, torch_pred = method_wrapper(*torch_args)
        conf_mats.append(torch_C)
        y_preds.append(torch_pred)

        # Run torch implementation on cuda
        if torch.cuda.is_available():
            torch_cuda_args = [
                torch.tensor(arg) if isinstance(arg, np.ndarray) else arg
                for arg in args
            ]
            torch_cuda_C, torch_cuda_pred = method_wrapper(*torch_cuda_args)
            conf_mats.append(torch_cuda_C)
            y_preds.append(torch_cuda_pred)

            assert torch_cuda_C == torch_C

        # Convert torch tensors to numpy arrays for easier comparison
        torch_C.tp = torch_C.tp.numpy()
        torch_C.fp = torch_C.fp.numpy()
        torch_C.fn = torch_C.fn.numpy()
        torch_C.tn = torch_C.tn.numpy()

        assert np_C_64 == torch_C

    return conf_mats, y_preds


@pytest.fixture(autouse=True, scope="session")
def test_method():
    return _test_method
