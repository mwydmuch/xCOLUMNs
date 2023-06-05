import numpy as np
import torch
from scipy.sparse import csr_matrix
from prediction_sparse import *
from random import randint
from tqdm import trange



FLOAT_TYPE=np.float32
IND_TYPE=np.int32


def predict_g(eta_pred, G):
    #return eta_pred.data * G[:,0][eta_pred.indices] + (eta_pred.data) * G[:,1][eta_pred.indices]
    #return eta_pred.data * G[:,0][eta_pred.indices] + (1.0 - eta_pred.data) * G[:,1][eta_pred.indices]
    return (eta_pred.data * (G[:,0][eta_pred.indices] - G[:,1][eta_pred.indices] - G[:,2][eta_pred.indices])) + G[:,1][eta_pred.indices]


def predict_top_k(eta_pred, G, k):
    """
    Predicts the labels for a given gradient matrix G and probability estimates eta_pred
    """
    #print("  Predicting top k based on marginal probabilities and gradient of confusion matrix")
    ni = eta_pred.shape[0]
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indices = np.zeros(ni * k, dtype=IND_TYPE)
    result_indptr = np.zeros(ni + 1, dtype=IND_TYPE)
    for i in trange(ni):
        eta_i = eta_pred[i]
        g = predict_g(eta_i, G)
        top_k = np.argpartition(-g, k)[:k]
        result_indices[i * k:(i + 1) * k] = sorted(eta_i.indices[top_k])
        result_indptr[i + 1] = result_indptr[i] + k

    return csr_matrix((result_data, result_indices, result_indptr), shape=(ni, G.shape[0]))


def calculate_confusion_matrix(pred_labels, true_labels, C_shape):
    """
    Calculate normalized confusion matrix
    """
    C = np.zeros(C_shape)
    C[:, 0] = calculate_etp(pred_labels, true_labels)
    C[:, 1] = calculate_efp(pred_labels, true_labels)
    C[:, 2] = calculate_efn(pred_labels, true_labels)
    C = C / true_labels.shape[0]
    
    return C


def calculate_loss(fn, C):
    C = torch.tensor(C, dtype=torch.float32)
    loss = fn(C)
    loss = torch.mean(loss)
    return float(loss)


def calculate_loss_with_gradient(fn, C, reg):
    #print("  Calculating loss with gradients")
    C = torch.tensor(C, requires_grad=True, dtype=torch.float32)
    loss = fn(C)
    loss = torch.mean(loss) #+ reg * torch.linalg.matrix_norm(C, ord='fro')
    loss.backward()
    return float(loss), np.array(C.grad)


def fw_macro_recall(C, epsilon=1e-5):
    return C[:,0] / (C[:,0] + C[:,2] + epsilon)


def fw_macro_precision(C, epsilon=1e-5):
    return C[:,0] / (C[:,0] + C[:,1] + epsilon)


def fw_macro_f1(C, beta=1.0, epsilon=1e-5):
    # precision = fw_macro_precision(C, epsilon=epsilon)
    # recall = fw_macro_recall(C, epsilon=epsilon)
    # return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)
    return 2 * C[:,0] / (2 * C[:,0] + C[:,1] + C[:,2] + epsilon)


def find_alpha(C, C_i, loss_func, g=1000):
    print("  Finding alpha")
    max_loss = 0
    max_alpha = 0

    for i in range(g):
        alpha = i / g
        new_C = (1 - alpha) * C + alpha * C_i
        loss = calculate_loss(loss_func, new_C)
        #print(f"    Alpha: {alpha}, loss: {loss * 100}")
        
        if loss > max_loss:
            max_loss = loss
            max_alpha = alpha

    return max_alpha


def frank_wolfe(y_true, eta_pred, max_iters=10, init="top", loss_func=None, k=5, stop_on_zero=True, reg=0, **kwargs):
    print("Starting Frank-Wolfe algorithm")

    m = eta_pred.shape[1]  # number of labels
    C_shape = (eta_pred.shape[1], 3)  # 0: TP, 1: FP, 2: FN
    init_G = np.zeros(C_shape)

    print(f"  Calculating initial loss based on {init}")
    if init == "top":
        init_G[:, 0] = 1
    elif init == "random":
        init_G[:, 0] = np.random.rand(m)
    init_pred = predict_top_k(eta_pred, init_G, k)
    #print("True:", y_true[0], "Pred:", init_pred[0])
    print("True:", y_true.shape, "Pred:", init_pred.shape, "Eta:", eta_pred.shape)
    C = calculate_confusion_matrix(init_pred, y_true, C_shape)
    loss = calculate_loss(loss_func, C)
    print(f"  Initial loss: {loss * 100}")
    
    classifiers = np.zeros((max_iters,) + C_shape)
    classifier_weights = np.zeros(max_iters)

    classifiers[0] = init_G  
    classifier_weights[0] = 1

    for i in range(1, max_iters):
        print(f"Starting iteration {i} ...")
        loss, G = calculate_loss_with_gradient(loss_func, C, reg)
        print(f"  Loss: {loss * 100}")
        # print(f"  Gradients: {G}")
        # print(f"  Grad sum {np.sum(G, axis=1)}")
        # print(f"  C matrix: {C}")
        
        classifiers[i] = G
        pred = predict_top_k(eta_pred, G, k)
        C_i = calculate_confusion_matrix(pred, y_true, C_shape)
        loss_i = calculate_loss(loss_func, C_i)
        print(f"  Loss_i: {loss_i * 100}")
        
        alpha = find_alpha(C, C_i, loss_func)
        #alpha = 1
        #alpha = 2 / (i + 2)
        
        classifier_weights[:i] *= (1 - alpha)
        classifier_weights[i] = alpha
        C = (1 - alpha) * C + alpha * C_i

        print(f"  Alpha: {alpha}")
        #print(f"  C_i matrix : {C_i}")
        #print(f"  new C matrix : {C}")

        # loss = calculate_loss(loss_func, C)
        # print(f"  Loss: {loss * 100}")
        # sampled_loss = sample_loss_from_classfiers(eta_pred, classifiers, classifier_weights, loss_func, y_true, C_shape, k=k, s=10)
        # print(f"  Sampled loss: {sampled_loss* 100}")

        if alpha == 0 and stop_on_zero:
            print("  Alpha is zero, stopping")
            classifiers = classifiers[:i]
            classifier_weights = classifier_weights[:i]
            break

    # Final loss calculation
    final_loss = calculate_loss(loss_func, C)
    print(f"  Final loss: {final_loss * 100}")

    # sampled_loss = sample_loss_from_classfiers(eta_pred, classifiers, classifier_weights, loss_func, y_true, C_shape, k=k)
    # print(f"  Final sampled loss: {sampled_loss* 100}")
    
    return classifiers, classifier_weights


def predict_top_k_for_classfiers(eta_pred, classifiers, classifier_weights, k=5, seed=0):
    if seed is not None:
        #print(f"  Using seed: {seed}")
        np.random.seed(seed)

    ni = eta_pred.shape[0]
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indices = np.zeros(ni * k, dtype=IND_TYPE)
    result_indptr = np.zeros(ni + 1, dtype=IND_TYPE)
    for i in trange(ni):
        c = np.random.choice(classifiers.shape[0], p=classifier_weights)
        G = classifiers[c]
        eta_i = eta_pred[i]
        g = predict_g(eta_i, G)
        top_k = np.argpartition(-g, k)[:k]
        result_indices[i * k:(i + 1) * k] = sorted(eta_i.indices[top_k])
        result_indptr[i + 1] = result_indptr[i] + k

    return csr_matrix((result_data, result_indices, result_indptr), shape=(ni, G.shape[0]))



def sample_loss_from_classfiers(eta_pred, classifiers, classifier_weights, loss_func, y_true, C_shape, k=5, s=5):
    losses = []
    for _ in range(s):
        classfiers_pred = predict_top_k_for_classfiers(eta_pred, classifiers, classifier_weights, k=k, seed=randint(0, 1000000))
        classfiers_C = calculate_confusion_matrix(classfiers_pred, y_true, C_shape)
        losses.append(calculate_loss(loss_func, classfiers_C))
    return np.mean(losses)
