import numpy as np
from prediction_utils import *
from tqdm import tqdm
from utils import *


def np_weighted_per_instance(prediction: np.ndarray, weights: np.ndarray, k: int = 5):
    ni, nl = prediction.shape
    assert weights.shape == (nl,)

    result = np.zeros((ni, nl), np.float32)

    for i in range(ni):
        eta = prediction[i, :]
        g = eta * weights
        top_k = np.argpartition(-g, k)[:k]
        result[i, top_k] = 1.0
    return result


def np_macro_population_cm_risk(probabilities: np.ndarray, k: int, measure_func: callable, 
                                greedy_start=False, tolerance: float = 1e-5, max_iter: int = 10, 
                                shuffle_order: bool =True, seed: int = None, **kwargs):
    ni, nl = probabilities.shape

    # Initialize the prediction variable with some feasible value
    result = random_at_k(probabilities, k)

    # For debug set it to first k labels
    #result = np.zeros(probabilities.shape, np.float32)
    #result[:,:k] = 1.0 

    # Other initializations
    # result = optimal_instance_precision(probabilities, k)
    # result = optimal_macro_recall(probabilities, k, marginal=np.mean(probabilities, axis=0))

    iters = 0
    for j in range(max_iter):

        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)
        
        if greedy_start and j == 0:
            Etp = np.zeros(nl, np.float32)
            Efp = np.zeros(nl, np.float32)
            Efn = np.zeros(nl, np.float32)
        else:
            # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
            Etp = np.sum(result * probabilities, axis=0)
            Efp = np.sum(result * (1-probabilities), axis=0)
            Efn = np.sum((1-result) * probabilities, axis=0)

        # Check expected conf matrices
        # print("Etp:", Etp.shape, type(Etp), Etp)
        # print("Efp:", Efp.shape, type(Efp), Efp)
        # print("Efn:", Efn.shape, type(Efn), Efn)

        old_score = np.mean(measure_func(Etp / ni, Efp / ni, Efn / ni))

        for i in order:
            eta = probabilities[i, :]

            # adjust a and b locally
            if not (greedy_start and j == 0):
                Etp -= result[i] * eta
                Efp -= result[i] * (1-eta)
                Efn -= (1-result[i]) * eta

            # calculate gain and selection
            Etpp = Etp + eta
            Efpp = Efp + (1-eta)
            Efnn = Efn + eta
            p_score = measure_func(Etpp / ni, Efpp / ni, Efn / ni)
            n_score = measure_func(Etp / ni, Efp / ni, Efnn / ni)
            gains = p_score - n_score
            top_k = np.argpartition(-gains, k)[:k]

            # update predictions
            result[i, :] = 0.0
            result[i, top_k] = 1.0

            # update a and b
            Etp += result[i] * eta
            Efp += result[i] * (1 - eta)
            Efn += (1 - result[i]) * eta

        iters = j
        new_score = np.mean(measure_func(Etp / ni, Efp / ni, Efn / ni))
        print(f"  Iteration {j + 1} finished, expected score: {old_score} -> {new_score}")
        if new_score <= old_score + tolerance:
            break

    return result, iters


def np_block_coordinate_coverage(probabilities: csr_matrix, k: int, greedy_start=False, tolerance: float = 1e-5, max_iter: int = 10, 
                                 shuffle_order: bool = True, seed: int = None, filename: str = None, **kwargs):
    """
    An efficient implementation of the block coordinate-descent for coverage
    """
    ni, nl = probabilities.shape

    # initialize the prediction variable with some feasible value
    result = predict_random_at_k(probabilities, k)
    probabilities = np.minimum(probabilities, 1 - 1e-5)
    iters = 0

    for j in range(max_iter):
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)
        
        if greedy_start and j == 0:
            f = np.ones(nl, np.float32)
        else:
            f = np.product(1 - result * probabilities, axis=0)
        old_cov = 1 - np.mean(f)

        for i in order:
            # adjust f locally
            f /= (1 - result[i] * probabilities[i])

            # calculate gain and selection
            g = f * probabilities[i]
            top_k = np.argpartition(-g, k)[:k]

            # update probabilities
            result[i, :] = 0.0
            result[i, top_k] = 1.0

            # update f
            f *= (1 - result[i] * probabilities[i])

        iters = j
        new_cov = 1 - np.mean(f)
        print(f"  Iteration {j + 1} finished, expected coverage: {old_cov} -> {new_cov}")
        if new_cov <= old_cov + tolerance:
            break
        
    return result, iters