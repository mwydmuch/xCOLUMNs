from typing import Union
import numpy as np
from scipy.sparse import csr_matrix
from bca_prediction import bca_with_0approx


def pu_through_etu(gt_valid: Union[np.ndarray, csr_matrix],
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
        result, _ = bca_with_0approx(current, k, utility_func=utility_func,
                                     tolerance=tolerance, seed=seed, verbose=False)
        pu_result[i, :] = result[-1, :]
    return pu_result
