from typing import Union
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import trange

from .block_coordinate import *


def pu_through_etu(y_proba: Union[np.ndarray, csr_matrix],
                   k: int,
                   utility_func: callable,
                   tolerance=1e-6,
                   seed: int = None,
                   gt_valid: Union[np.ndarray, csr_matrix] = None,
                   **kwargs,
                   ):
    """
    :param gt_valid: Ground-truth labels for validation set
    :param pred_test: Predicted probabilities on the test set
    :param k: Number of predictions per instance
    :param utility_func: Which metric to optimize for
    :param tolerance: Tolerance for the BCA inference
    :param seed: Seed for the BCA inference
    """

    # TODO: version of this idea that takes etas on a validation set instead of ground-truth on the training set
    # TODO: instead of adding the new example using "eta", sample possible labels for the new example, and produce a
    #       distribution over predictions
    # TODO: approximate inference, e.g., just calculate optimal confusion matrix on gt_valid, and *only* perform inference
    #       on the new sample like in the greedy algorithm.

    pu_result = np.zeros_like(y_proba)
    print(y_proba.shape, gt_valid.shape, y_proba[0:0 + 1, :].shape, type(y_proba), type(gt_valid), type(y_proba[0:0 + 1, :]))
    for i in trange(y_proba.shape[0]):
        current = np.concatenate((gt_valid, y_proba[i:i + 1, :]), axis=0)
        result = bc_with_0approx(current, k, utility_func=utility_func,
                                     tolerance=tolerance, seed=seed, verbose=False)
        pu_result[i, :] = result[-1, :]
    return pu_result


def pu_through_etu_macro_f1(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return pu_through_etu(
        y_proba, k=k, utility_func=macro_fmeasure_on_conf_matrix, **kwargs
    )


def pu_through_etu2(gt_valid: Union[np.ndarray, csr_matrix],
                   pred_test: Union[np.ndarray, csr_matrix],
                   k: int,
                   utility_func: callable,
                   tolerance=1e-6,
                   seed: int = None,
                   ):
    """
    :param gt_valid: Ground-truth labels for validation set
    :param pred_test: Predicted probabilities on the test set
    :param k: Number of predictions per instance
    :param utility_func: Which metric to optimize for
    :param tolerance: Tolerance for the BCA inference
    :param seed: Seed for the BCA inference
    """

    # TODO: version of this idea that takes etas on a validation set instead of ground-truth on the training set
    # TODO: instead of adding the new example using "eta", sample possible labels for the new example, and produce a
    #       distribution over predictions
    # TODO: approximate inference, e.g., just calculate optimal confusion matrix on gt_valid, and *only* perform inference
    #       on the new sample like in the greedy algorithm.

    pu_result = np.zeros_like(pred_test)
    for i in range(len(pred_test)):
        current = np.concatenate([gt_valid, pred_test[i:i + 1, :]])
        result, _ = bc_with_0approx(current, k, utility_func=utility_func,
                                     tolerance=tolerance, seed=seed, verbose=False)
        pu_result[i, :] = result[-1, :]
    return pu_result