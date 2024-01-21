import random
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import trange

from .block_coordinate import *
from .weighted_prediction import *
from .find_classifier_frank_wolfe import find_classifier_frank_wolfe, macro_f1_C


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
    y_true_valid: Union[np.ndarray, csr_matrix] = None,
    return_meta: bool = False,
    use_true_valid_c: bool = False,
    use_true_valid_bc: bool = False,
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

    print(f"  Running BC {num_valid_sets} times with sets of {valid_set_size} size")

    classifiers = []
    for i in range(num_valid_sets):
        if valid_set_size != 1 and isinstance(valid_set_size, float):
            print(f"    Creating subset")
            selected_rows = np.random.choice(
                    y_proba_valid.shape[0], int(y_proba_valid.shape[0] * valid_set_size)
                )
            i_y_proba_valid = y_proba_valid[selected_rows]
            i_y_true_valid = y_true_valid[selected_rows]
        elif valid_set_size > 1 and isinstance(valid_set_size, int):
            print(f"    Creating subset")
            selected_rows = np.random.choice(y_proba_valid.shape[0], valid_set_size)
            i_y_proba_valid = y_proba_valid[selected_rows]
            i_y_true_valid = y_true_valid[selected_rows]
        else:
            i_y_proba_valid = y_proba_valid
            i_y_true_valid = y_true_valid

        if use_true_valid_bc:
            i_y_proba_valid = i_y_true_valid

        print(f"  Running BC for {i} subset of the validation set")
        y_pred = bc_with_0approx(
            i_y_proba_valid,
            k,
            bin_utility_func=bin_utility_func,
            tolerance=tolerance,
            seed=seed,
            verbose=True,
        )
        if use_true_valid_bc or use_true_valid_c:
            tp, fp, fn, tn = calculate_confusion_matrix(i_y_true_valid, y_pred)
        else:
            tp, fp, fn, tn = calculate_confusion_matrix(i_y_proba_valid, y_pred)
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



def online_bc_with_fw(
    y_proba: Union[np.ndarray, csr_matrix],
    k: int,
    bin_utility_func_fw: callable,
    bin_utility_func_bc: callable,
    tolerance=1e-6,
    seed: int = None,
    greedy: bool = False,
    num_valid_sets=10,
    valid_set_size=0.9,
    y_proba_valid: Union[np.ndarray, csr_matrix] = None,
    y_true_valid: Union[np.ndarray, csr_matrix] = None,
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

    print(f"  Running FW {num_valid_sets} times with sets of {valid_set_size} size")

    classifiers = []
    for i in range(num_valid_sets):
        if valid_set_size != 1 and isinstance(valid_set_size, float):
            print(f"    Creating subset")
            selected_rows = np.random.choice(
                    y_proba_valid.shape[0], int(y_proba_valid.shape[0] * valid_set_size)
                )
            i_y_proba_valid = y_proba_valid[selected_rows]
            i_y_true_valid = y_true_valid[selected_rows]
        elif valid_set_size > 1 and isinstance(valid_set_size, int):
            print(f"    Creating subset")
            selected_rows = np.random.choice(y_proba_valid.shape[0], valid_set_size)
            i_y_proba_valid = y_proba_valid[selected_rows]
            i_y_true_valid = y_true_valid[selected_rows]
        else:
            i_y_proba_valid = y_proba_valid
            i_y_true_valid = y_true_valid

        print(f"  Running FW for {i + 1}/{num_valid_sets} subset of the validation set")
        _classifiers, _classifier_weights, _meta = find_classifier_frank_wolfe(
            i_y_true_valid, i_y_proba_valid, bin_utility_func_fw, max_iters=20, k=k, return_meta=True, **kwargs
        )
        valid_n, _ = y_true_valid.shape
        C = _meta["C"]
        tp = C[:,0] * valid_n
        fp = C[:,1] * valid_n
        fn = C[:,2] * valid_n
        tn = C[:,3] * valid_n
        #print(tp.shape, fp.shape, fn.shape, tn.shape)
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
            bin_utility_func_bc,
            only_pred=(not greedy),
            greedy=greedy,
        )

    if return_meta:
        meta["time"] = time() - meta["time"]
        return y_pred, meta
    else:
        return y_pred


def online_bc_with_fw_macro_f1(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return online_bc_with_fw(
        y_proba, k=k, bin_utility_func_fw=macro_f1_C, bin_utility_func_bc=macro_fmeasure_on_conf_matrix, **kwargs
    )


def online_with_feedback():
    pass


def online_with_feedback_f1(y_proba: Union[np.ndarray, csr_matrix], k: int = 5, **kwargs):
    return online_with_feedback(
        y_proba, k=k, bin_utility_func=macro_fmeasure_on_conf_matrix, **kwargs
    )