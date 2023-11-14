import numpy as np
import torch
from scipy.sparse import csr_matrix
from utils_sparse import *
from random import randint

from typing import Union


FLOAT_TYPE = np.float32
IND_TYPE = np.int32
EPS = 1e-6


def select_top_k_csr(y_proba, G, k):
    # True negatives are not used in the utility function, so we can ignore them here
    u = (
        (
            y_proba.data
            * (
                G[:, 0][y_proba.indices]
                - G[:, 1][y_proba.indices]
                - G[:, 2][y_proba.indices]
                + G[:, 3][y_proba.indices]
            )
        )
        + G[:, 1][y_proba.indices]
        - G[:, 3][y_proba.indices]
    )
    top_k = np.argpartition(-u, k)[:k]
    return top_k


def select_top_k_np(y_proba, G, k):
    # True negatives are not used in the utility function, so we can ignore them here
    u = (y_proba * (G[:, 0] - G[:, 1] - G[:, 2] + G[:, 3])) + G[:, 1] - G[:, 3]
    top_k = np.argpartition(-u, k)[:k]
    return top_k


def predict_top_k_csr(y_proba, G, k):
    """
    Predicts the labels for a given gradient matrix G and probability estimates y_proba in dense format
    """
    ni = y_proba.shape[0]
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indices = np.zeros(ni * k, dtype=IND_TYPE)
    result_indptr = np.zeros(ni + 1, dtype=IND_TYPE)
    for i in range(ni):
        eta_i = y_proba[i]
        top_k = select_top_k_csr(eta_i, G, k)
        result_indices[i * k : (i + 1) * k] = sorted(eta_i.indices[top_k])
        result_indptr[i + 1] = result_indptr[i] + k

    return csr_matrix(
        (result_data, result_indices, result_indptr), shape=(ni, G.shape[0])
    )


def predict_top_k_np(y_proba, G, k):
    """
    Predicts the labels for a given gradient matrix G and probability estimates y_proba in sparse format
    """
    ni = y_proba.shape[0]
    result = np.zeros(y_proba.shape, dtype=FLOAT_TYPE)
    for i in range(ni):
        eta_i = y_proba[i]
        top_k = select_top_k_np(eta_i, G, k)
        result[i, top_k] = 1.0

    return result


def predict_top_k(y_proba, G, k):
    """
    Predicts the labels for a given gradient matrix G and probability estimates y_proba
    """
    if isinstance(y_proba, np.ndarray):
        return predict_top_k_np(y_proba, G, k)
    elif isinstance(y_proba, csr_matrix):
        return predict_top_k_csr(y_proba, G, k)


def calculate_confusion_matrix_csr(y_true, y_pred, C_shape):
    """
    Calculate normalized confusion matrix for true labels and predicted labels in sparse format
    """
    # True negatives are not used in the utility function, so we can ignore them here
    C = np.zeros(C_shape)
    C[:, 0] = calculate_tp_csr(y_true, y_pred)
    C[:, 1] = calculate_fp_csr(y_true, y_pred)
    C[:, 2] = calculate_fn_csr(y_true, y_pred)
    C = C / y_true.shape[0]

    return C


def calculate_confusion_matrix_np(y_true, y_pred, C_shape):
    """
    Calculate normalized confusion matrix for true labels and predicted labels in dense format
    """
    # True negatives are not used in the utility function, so we can ignore them here
    # C = np.zeros((y_true.shape[1], 3))
    C = np.zeros(C_shape)
    C[:, 0] = np.sum(y_pred * y_true, axis=0)
    C[:, 1] = np.sum(y_pred * (1 - y_true), axis=0)
    C[:, 2] = np.sum((1 - y_pred) * y_true, axis=0)
    C = C / y_true.shape[0]

    return C


def calculate_utility(fn, C):
    C = torch.tensor(C, dtype=torch.float32)
    utility = fn(C)
    utility = torch.mean(utility)
    return float(utility)


def calculate_utility_with_gradient(fn, C):
    C = torch.tensor(C, requires_grad=True, dtype=torch.float32)
    utility = fn(C)
    utility = torch.mean(utility)
    utility.backward()
    return float(utility), np.array(C.grad)


def find_best_alpha(C, C_i, utility_func, search_step=0.001):
    max_utility = 0
    max_alpha = 0

    for alpha in np.arange(0, 1, search_step):
        new_C = (1 - alpha) * C + alpha * C_i
        utility = calculate_utility(utility_func, new_C)

        if utility > max_utility:
            max_utility = utility
            max_alpha = alpha

    return max_alpha


def find_classifier_frank_wolfe(
    y_true: Union[np.ndarray, csr_matrix],
    y_proba: Union[np.ndarray, csr_matrix],
    utility_func,
    max_iters: int = 20,
    init: str = "topk",
    k: int = 5,
    use_best_alpha: bool = True,
    stop_on_alpha_zero: bool = True,
    alpha_search_step: float = 0.001,
    verbose: bool = True,
    **kwargs,
):
    log = print
    if not verbose:
        log = lambda *args, **kwargs: None

    if isinstance(y_true, np.ndarray) and isinstance(y_proba, np.ndarray):
        func_calculate_confusion_matrix = calculate_confusion_matrix_np
        func_predict_top_k = predict_top_k_np
    elif isinstance(y_true, csr_matrix) and isinstance(y_proba, csr_matrix):
        func_calculate_confusion_matrix = calculate_confusion_matrix_csr
        func_predict_top_k = predict_top_k_csr
    else:
        raise ValueError(
            f"y_true and y_proba have unsuported combination of types {type(y_true)}, {type(y_proba)}"
        )

    log("Starting Frank-Wolfe algorithm")
    m = y_proba.shape[1]  # number of labels
    C_shape = (y_proba.shape[1], 4)  # 0: TP, 1: FP, 2: FN, #3: TN
    init_G = np.zeros(C_shape)

    log(f"  Calculating initial utility based on {init} predictions ...")
    if init == "topk":
        init_G[:, 0] = 1
    elif init == "random":
        init_G[:, 0] = np.random.rand(m)
    init_pred = func_predict_top_k(y_proba, init_G, k)
    log(
        f"    y_true: {y_true.shape}, y_pred: {init_pred.shape}, y_proba: {y_proba.shape}"
    )
    C = func_calculate_confusion_matrix(y_true, init_pred, C_shape=C_shape)
    utility = calculate_utility(utility_func, C)
    log(f"    initial utility: {utility}")

    classifiers = np.zeros((max_iters,) + C_shape)
    classifier_weights = np.zeros(max_iters)

    classifiers[0] = init_G
    classifier_weights[0] = 1

    meta = {"alphas": [], "utilities": []}

    for i in range(1, max_iters):
        log(f"  Starting iteration {i} ...")
        utility, G = calculate_utility_with_gradient(utility_func, C)
        meta["utilities"].append(utility)

        # log(f"    prev C matrix = {C}")
        log(f"    utility = {utility}")
        # log(f"    gradients = {G}")
        #log(f"    new a = {G[:,0] - G[:,1] - G[:,2] + G[:, 3]}")
        #log(f"    new b = {G[:,1] - G[:, 3]}")

        classifiers[i] = G
        y_pred = func_predict_top_k(y_proba, G, k)
        C_i = func_calculate_confusion_matrix(y_true, y_pred, C_shape=C_shape)
        utility_i = calculate_utility(utility_func, C_i)

        if use_best_alpha:
            alpha = find_best_alpha(C, C_i, utility_func, search_step=alpha_search_step)
        else:
            alpha = 2 / (i + 1)
        meta["alphas"].append(alpha)

        log(f"    utility_i = {utility_i}")
        log(f"    alpha = {alpha}")

        classifier_weights[:i] *= 1 - alpha
        classifier_weights[i] = alpha
        C = (1 - alpha) * C + alpha * C_i

        # log(f"  C_i matrix : {C_i}")
        # log(f"  new C matrix : {C}")

        if alpha == 0 and stop_on_alpha_zero:
            log("    alpha is 0, stopping")
            classifiers = classifiers[:i]
            classifier_weights = classifier_weights[:i]
            break

    meta["iters"] = i

    # Final utility calculation
    final_utility = calculate_utility(utility_func, C)
    log(f"  Final utility: {final_utility}, number of iterations: {i}")

    # sampled_utility = sample_utility_from_classfiers(y_proba, classifiers, classifier_weights, utility_func, y_true, C_shape, k=k)
    # print(f"  Final sampled utility: {sampled_utility* 100}")

    return classifiers, classifier_weights, meta


def predict_top_k_for_classfiers_csr(
    y_proba, classifiers, classifier_weights, k=5, seed=0
):
    if seed is not None:
        np.random.seed(seed)

    ni = y_proba.shape[0]
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indices = np.zeros(ni * k, dtype=IND_TYPE)
    result_indptr = np.zeros(ni + 1, dtype=IND_TYPE)
    for i in range(ni):
        c = np.random.choice(classifiers.shape[0], p=classifier_weights)
        G = classifiers[c]
        eta_i = y_proba[i]
        top_k = select_top_k_csr(eta_i, G, k)
        result_indices[i * k : (i + 1) * k] = sorted(eta_i.indices[top_k])
        result_indptr[i + 1] = result_indptr[i] + k

    return csr_matrix(
        (result_data, result_indices, result_indptr), shape=(ni, G.shape[0])
    )


def predict_top_k_for_classfiers_np(
    y_proba, classifiers, classifier_weights, k=5, seed=0
):
    if seed is not None:
        np.random.seed(seed)

    ni = y_proba.shape[0]
    result = np.zeros(y_proba.shape, dtype=FLOAT_TYPE)
    for i in range(ni):
        c = np.random.choice(classifiers.shape[0], p=classifier_weights)
        G = classifiers[c]
        eta_i = y_proba[i]
        top_k = select_top_k_np(eta_i, G, k)
        result[i, top_k] = 1.0

    return result


def predict_top_k_for_classfiers(y_proba, classifiers, classifier_weights, k=5, seed=0):
    if isinstance(y_proba, np.ndarray):
        return predict_top_k_for_classfiers_np(
            y_proba, classifiers, classifier_weights, k=k, seed=seed
        )
    elif isinstance(y_proba, csr_matrix):
        return predict_top_k_for_classfiers_csr(
            y_proba, classifiers, classifier_weights, k=k, seed=seed
        )
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")


def sample_utility_from_classfiers_csr(
    y_proba, classifiers, classifier_weights, utility_func, y_true, C_shape, k=5, s=5
):
    utilities = []
    for _ in range(s):
        classfiers_pred = predict_top_k_for_classfiers_csr(
            y_proba, classifiers, classifier_weights, k=k, seed=randint(0, 1000000)
        )
        classfiers_C = calculate_confusion_matrix_csr(classfiers_pred, y_true, C_shape)
        utilities.append(calculate_utility(utility_func, classfiers_C))
    return np.mean(utilities)


# Utility functions defined using PyTorch
def macro_jaccard_C(C, epsilon=EPS):
    return C[:, 0] / (C[:, 0] + C[:, 1] + C[:, 2] + epsilon)


def macro_sqrt_tp_C(C, epsilon=EPS):
    return torch.sqrt(C[:,0] + epsilon)


def precision_at_k_C(C, k=5):
    return C[:, 0] / k


def macro_recall_C(C, epsilon=EPS):
    return C[:, 0] / (C[:, 0] + C[:, 2] + epsilon)


def macro_precision_C(C, epsilon=EPS):
    return C[:, 0] / (C[:, 0] + C[:, 1] + epsilon)


def macro_f1_C(C, epsilon=EPS):
    return 2 * C[:, 0] / (2 * C[:, 0] + C[:, 1] + C[:, 2] + epsilon)


def balanced_acc_C(C):
    return (C[:, 0] / (C[:, 0] + C[:, 2])) * (C[:, 0] / (C[:, 0] + C[:, 1]))


def mixed_instance_prec_macro_prec_C(C, alpha=0.001, epsilon=EPS):
    return (1 - alpha) * precision_at_k_C(C) + alpha * macro_precision_C(C)


def mixed_instance_prec_macro_f1_C(C, alpha=0.9, epsilon=EPS):
    return (1 - alpha) * precision_at_k_C(C) + alpha * macro_f1_C(C)