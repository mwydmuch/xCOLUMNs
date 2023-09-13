import numpy as np


def _make_filler(num_labels: int, k: int):
    all_labels = set(range(num_labels))

    def filler(target: np.ndarray, already_selected: np.ndarray):
        free_labels = list(all_labels - set(already_selected))
        additional = np.random.choice(
            free_labels, k - len(already_selected), replace=False
        )
        target[already_selected] = 1.0
        target[additional] = 1.0

    return filler


def predict_random_at_k(y_proba: np.ndarray, k: int = 5):
    """
    Randomly select among the true labels. A very simple baseline
    that requires 0/1 inputs.
    """
    ni, nl = y_proba.shape
    result = np.zeros_like(y_proba)
    filler = _make_filler(nl, k)
    for i in range(ni):
        lbl_at_i = np.nonzero(y_proba[i, :])[0]
        if len(lbl_at_i) > k:
            active = np.random.choice(lbl_at_i, k, replace=False)
            result[i, active] = 1.0
        else:
            filler(result[i, :], lbl_at_i)
    return result


def random_at_k(y_proba: np.ndarray, k: int = 5):
    """
    Select predicted labels completely randomly, ignoring whether they are true/false.
    """
    ni, nl = y_proba.shape
    result = np.zeros_like(y_proba)
    all_labels = list(range(nl))
    for i in range(ni):
        additional = np.random.choice(all_labels, k, replace=False)
        result[i, additional] = 1.0
    return result
