import random
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import trange

from .block_coordinate import *
from .weighted_prediction import *


def pu_through_etu(
    y_proba: Union[np.ndarray, csr_matrix],
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
    print(
        y_proba.shape,
        gt_valid.shape,
        y_proba[0 : 0 + 1, :].shape,
        type(y_proba),
        type(gt_valid),
        type(y_proba[0 : 0 + 1, :]),
    )
    for i in trange(y_proba.shape[0]):
        current = np.concatenate((gt_valid, y_proba[i : i + 1, :]), axis=0)
        result = bc_with_0approx(
            current,
            k,
            utility_func=utility_func,
            tolerance=tolerance,
            seed=seed,
            verbose=False,
        )
        pu_result[i, :] = result[-1, :]
    return pu_result


def online_bc_macro_f1(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return online_bc(
        y_proba, k=k, bin_utility_func=macro_fmeasure_on_conf_matrix, **kwargs
    )


def online_bc(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    bin_utility_func: callable,
    tolerance=1e-6,
    seed: int = None,
    greedy: bool = False,
    num_valid_sets=10,
    valid_set_size=0.9,
    y_proba_valid: Union[np.ndarray, csr_matrix] = None,
    return_meta: bool = False,
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

    # Initialize the meta data dictionary
    meta = {"iters": 1, "valid_iters": num_valid_sets, "time": time()}

    n, m = y_proba.shape
    # Get specialized functions
    if isinstance(y_proba, np.ndarray):
        bc_with_0approx_step_func = bc_with_0approx_np_step
    elif isinstance(y_proba, csr_matrix):
        bc_with_0approx_step_func = bc_with_0approx_csr_step
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")

    random.seed(seed)

    y_valid = y_proba_valid

    classifiers = []
    for i in range(num_valid_sets):
        if valid_set_size != 1 and isinstance(y_valid, float):
            i_y_valid = y_valid[
                np.random.choice(
                    y_valid.shape[0], int(y_valid.shape[0] * valid_set_size)
                )
            ]
        elif valid_set_size > 1 and isinstance(y_valid, int):
            i_y_valid = y_valid[np.random.choice(y_valid.shape[0], valid_set_size)]
        else:
            i_y_valid = y_valid

        print(f"  Running BC for {i} subset of the validation set")
        y_pred = bc_with_0approx(
            i_y_valid,
            k,
            bin_utility_func=bin_utility_func,
            tolerance=tolerance,
            seed=seed,
            verbose=True,
        )
        tp, fp, fn, tn = calculate_confusion_matrix(i_y_valid, y_pred)
        classifiers.append((tp, fp, fn, tn))

    print("  Predicting the test set")
    y_pred = predict_top_k(y_proba, k, return_meta=False)
    for i in trange(n):
        tp, fp, fn, tn = random.choice(classifiers)
        bc_with_0approx_step_func(
            y_proba,
            y_pred,
            i,
            tp,
            fp,
            fn,
            tn,
            k,
            bin_utility_func,
            only_pred=(not greedy),
            greedy=greedy,
        )

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred
