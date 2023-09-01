import numpy as np
import torch
from scipy.sparse import csr_matrix
from utils_sparse import *
from random import randint
from tqdm import trange
from typing import Union


FLOAT_TYPE=np.float32
IND_TYPE=np.int32
EPS = 1e-5


def precision_C(C, k=5):
    return C[:,0] / k


def macro_recall_C(C, epsilon=EPS):
    return C[:,0] / (C[:,0] + C[:,2] + epsilon)


def macro_precision_C(C, epsilon=EPS):
    return C[:,0] / (C[:,0] + C[:,1] + epsilon)


def macro_f1_C(C, epsilon=EPS):
    return 2 * C[:,0] / (2 * C[:,0] + C[:,1] + C[:,2] + epsilon)


def select_top_k_csr(y_proba, G, k):
    # True negatives are not used in the utility function, so we can ignore them here
    u = (y_proba.data * (G[:,0][y_proba.indices] - G[:,1][y_proba.indices] - G[:,2][y_proba.indices])) + G[:,1][y_proba.indices]
    top_k = np.argpartition(-u, k)[:k]
    return top_k


def select_top_k_np(y_proba, G, k):
    # True negatives are not used in the utility function, so we can ignore them here
    u = (y_proba (G[:,0] - G[:,1] - G[:,2])) + G[:,1]
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
    for i in trange(ni):
        eta_i = y_proba[i]
        top_k = select_top_k_csr(eta_i, G, k)
        result_indices[i * k:(i + 1) * k] = sorted(eta_i.indices[top_k])
        result_indptr[i + 1] = result_indptr[i] + k

    return csr_matrix((result_data, result_indices, result_indptr), shape=(ni, G.shape[0]))


def predict_top_k_np(y_proba, G, k):
    """
    Predicts the labels for a given gradient matrix G and probability estimates y_proba in sparse format
    """
    ni = y_proba.shape[0]
    result = np.zeros(y_proba.shape, dtype=FLOAT_TYPE)
    for i in trange(ni):
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


def calculate_confusion_matrix_np(y_true, y_pred):
    """
    Calculate normalized confusion matrix for true labels and predicted labels in dense format
    """
    # True negatives are not used in the utility function, so we can ignore them here
    C = np.zeros((y_true.shape[1], 3))
    C[:, 0] = np.sum(y_pred * y_true, axis=0)
    C[:, 1] = np.sum(y_pred * (1-y_true), axis=0)
    C[:, 2] = np.sum((1-y_pred) * y_true, axis=0)
    C = C / y_true.shape[0]
    
    return C


def calculate_utility(fn, C):
    C = torch.tensor(C, dtype=torch.float32)
    utility = fn(C)
    utility = torch.mean(utility)
    return float(utility)


def calculate_utility_with_gradient(fn, C, reg):
    #print("  Calculating utility with gradients")
    C = torch.tensor(C, requires_grad=True, dtype=torch.float32)
    utility = fn(C)
    utility = torch.mean(utility) #+ reg * torch.linalg.matrix_norm(C, ord='fro')
    utility.backward()
    return float(utility), np.array(C.grad)


def find_alpha(C, C_i, utility_func, g=1000):
    print("  Finding alpha")
    max_utility = 0
    max_alpha = 0

    for i in range(g):
        alpha = i / g
        new_C = (1 - alpha) * C + alpha * C_i
        utility = calculate_utility(utility_func, new_C)
        #print(f"    Alpha: {alpha}, utility: {utility * 100}")
        
        if utility > max_utility:
            max_utility = utility
            max_alpha = alpha

    return max_alpha


def frank_wolfe(y_true: Union[np.ndarray, csr_matrix], y_proba: Union[np.ndarray, csr_matrix], utility_func, max_iters: int = 10, init: str = "topk", k=5, stop_on_zero=True, reg=0, **kwargs):
    if isinstance(y_true, np.ndarray) and isinstance(y_proba, np.ndarray):
        func_calculate_confusion_matrix = calculate_confusion_matrix_np
        func_predict_top_k = predict_top_k_np
    elif isinstance(y_true, csr_matrix) and isinstance(y_proba, csr_matrix):
        func_calculate_confusion_matrix = calculate_confusion_matrix_csr
        func_predict_top_k = predict_top_k_csr
    else:
        raise ValueError(f"y_true and y_proba have unsuported combination of types {type(y_true)}, {type(y_proba)}")
    

    print("Starting Frank-Wolfe algorithm")
    m = y_proba.shape[1]  # number of labels
    C_shape = (y_proba.shape[1], 3)  # 0: TP, 1: FP, 2: FN
    init_G = np.zeros(C_shape)

    print(f"  Calculating initial utility based on {init} predictions ...")
    if init == "topk":
        init_G[:, 0] = 1
    elif init == "random":
        init_G[:, 0] = np.random.rand(m)
    init_pred = func_predict_top_k(y_proba, init_G, k)
    #print("True:", y_true[0], "Pred:", init_pred[0])
    print(f"  y_true: {y_true.shape}, y_pred: {init_pred.shape}, y_proba: {y_proba.shape}")
    C = func_calculate_confusion_matrix(y_true, init_pred, C_shape)
    utility = calculate_utility(utility_func, C)
    print(f"  Initial utility: {utility * 100}")
    
    classifiers = np.zeros((max_iters,) + C_shape)
    classifier_weights = np.zeros(max_iters)

    classifiers[0] = init_G  
    classifier_weights[0] = 1

    meta = {"alphas": [], "utilities": []}

    for i in range(1, max_iters):
        print(f"Starting iteration {i} ...")
        utility, G = calculate_utility_with_gradient(utility_func, C, reg)
        meta["utilities"].append(utility)
        print(f"  utility: {utility * 100}")

        # print(f"  Gradients: {G}")
        # print(f"  Grad sum {np.sum(G, axis=1)}")
        # print(f"  C matrix: {C}")
        
        classifiers[i] = G
        y_pred = func_predict_top_k(y_proba, G, k)
        C_i = func_calculate_confusion_matrix(y_true, y_pred, C_shape)
        utility_i = calculate_utility(utility_func, C_i)
        print(f"  utility_i: {utility_i * 100}")
        
        alpha = find_alpha(C, C_i, utility_func)
        meta["alphas"].append(alpha)
        print(f"  alpha: {alpha}")
        
        classifier_weights[:i] *= (1 - alpha)
        classifier_weights[i] = alpha
        C = (1 - alpha) * C + alpha * C_i

        #print(f"  C_i matrix : {C_i}")
        #print(f"  new C matrix : {C}")

        # utility = calculate_utility(utility_func, C)
        # print(f"  utility: {utility * 100}")
        # sampled_utility = sample_utility_from_classfiers(y_proba, classifiers, classifier_weights, utility_func, y_true, C_shape, k=k, s=10)
        # print(f"  Sampled utility: {sampled_utility* 100}")

        if alpha == 0 and stop_on_zero:
            print("  Alpha is zero, stopping")
            classifiers = classifiers[:i]
            classifier_weights = classifier_weights[:i]
            break

    meta["iters"] = i

    # Final utility calculation
    final_utility = calculate_utility(utility_func, C)
    print(f"  Final utility: {final_utility * 100}")

    # sampled_utility = sample_utility_from_classfiers(y_proba, classifiers, classifier_weights, utility_func, y_true, C_shape, k=k)
    # print(f"  Final sampled utility: {sampled_utility* 100}")
    
    return classifiers, classifier_weights, meta


def predict_top_k_for_classfiers_csr(y_proba, classifiers, classifier_weights, k=5, seed=0):
    if seed is not None:
        #print(f"  Using seed: {seed}")
        np.random.seed(seed)

    ni = y_proba.shape[0]
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indices = np.zeros(ni * k, dtype=IND_TYPE)
    result_indptr = np.zeros(ni + 1, dtype=IND_TYPE)
    for i in trange(ni):
        c = np.random.choice(classifiers.shape[0], p=classifier_weights)
        G = classifiers[c]
        eta_i = y_proba[i]
        top_k = select_top_k_csr(eta_i, G, k)
        result_indices[i * k:(i + 1) * k] = sorted(eta_i.indices[top_k])
        result_indptr[i + 1] = result_indptr[i] + k

    return csr_matrix((result_data, result_indices, result_indptr), shape=(ni, G.shape[0]))


def predict_top_k_for_classfiers_np(y_proba, classifiers, classifier_weights, k=5, seed=0):
    if seed is not None:
        #print(f"  Using seed: {seed}")
        np.random.seed(seed)

    ni = y_proba.shape[0]
    result = np.zeros(y_proba.shape, dtype=FLOAT_TYPE)
    for i in trange(ni):
        c = np.random.choice(classifiers.shape[0], p=classifier_weights)
        G = classifiers[c]
        eta_i = y_proba[i]
        top_k = select_top_k_np(eta_i, G, k)
        result[i, top_k] = 1.0

    return result


def predict_top_k_for_classfiers(y_proba, classifiers, classifier_weights, k=5, seed=0):
    if isinstance(y_proba, np.ndarray):
        return predict_top_k_for_classfiers_np(y_proba, classifiers, classifier_weights, k=k, seed=seed)
    elif isinstance(y_proba, csr_matrix):
        return predict_top_k_for_classfiers_csr(y_proba, classifiers, classifier_weights, k=k, seed=seed)
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")


def sample_utility_from_classfiers_csr(y_proba, classifiers, classifier_weights, utility_func, y_true, C_shape, k=5, s=5):
    utilities = []
    for _ in range(s):
        classfiers_pred = predict_top_k_for_classfiers_csr(y_proba, classifiers, classifier_weights, k=k, seed=randint(0, 1000000))
        classfiers_C = calculate_confusion_matrix_csr(classfiers_pred, y_true, C_shape)
        utilities.append(calculate_utility(utility_func, classfiers_C))
    return np.mean(utilities)
