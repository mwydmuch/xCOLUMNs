import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from numba import njit
from tqdm import tqdm
from utils import *


FLOAT_TYPE=np.float64


@njit
def numba_first_first_k(ni, k):
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indicies = np.zeros(ni * k, dtype=np.int32)
    result_indptr = np.zeros(ni + 1, dtype=np.int32)
    for i in range(ni):
        result_indicies[(i * k):((i + 1) * k)] = np.arange(0, k, 1, FLOAT_TYPE)
        result_indptr[i + 1] = result_indptr[i] + k
    return result_data, result_indicies, result_indptr


@njit
def numba_random_at_k(data: np.ndarray, indicies: np.ndarray, indptr: np.ndarray, 
                     ni: int, nl: int, k: int,):
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indicies = np.zeros(ni * k, dtype=np.int32)
    result_indptr = np.zeros(ni + 1, dtype=np.int32)
    for i in range(ni):
        row_indicies = indicies[indptr[i]:indptr[i+1]]
        result_indicies[i * k:(i + 1) * k] = np.random.choice(row_indicies, k, replace=False)
        result_indptr[i + 1] = result_indptr[i] + k

    return result_data, result_indicies, result_indptr


#@njit
def argtopk(data, indices, k):
    """
    Returns the indices of the top k elements
    """
    #return indices[np.argsort(-data)[:k]]
    # argpartition is faster than sort for large datasets, but not supported by Numba
    return indices[np.argpartition(-data, k)[:k]]



def csr_weighted_per_instance(prediction: csr_matrix, weights: np.ndarray, k: int = 5):
    # Since many numpy functions are not supported for sparse matrices,
    ni, nl = prediction.shape
    data, indices, indptr = numba_weighted_per_instance(prediction.data, prediction.indices, prediction.indptr, weights, ni, nl, k)
    return csr_matrix((data, indices, indptr), shape=prediction.shape)


#@njit
def numba_weighted_per_instance(data: np.ndarray, indicies: np.ndarray, indptr: np.ndarray, 
                                weights: np.ndarray, ni: int, nl: int, k: int):
    result_data = np.ones(ni * k, dtype=FLOAT_TYPE)
    result_indicies = np.zeros(ni * k, dtype=np.int32)
    result_indptr = np.zeros(ni + 1, dtype=np.int32)

    # This can be done in parallel, but Numba parallelism seems to not work well here
    for i in range(ni):
        row_data = data[indptr[i]:indptr[i+1]]
        row_indicies = indicies[indptr[i]:indptr[i+1]]
        row_weights = weights[row_indicies].reshape(-1) * row_data
        top_k = argtopk(row_weights, row_indicies, k)
        result_indicies[i * k:(i + 1) * k] = top_k
        result_indptr[i + 1] = result_indptr[i] + k

    return result_data, result_indicies, result_indptr


def calculate_etp(result: csr_matrix, probabilities: csr_matrix):
    return (result.multiply(probabilities)).sum(axis=0)


# This is a bit slow, TODO: make it faster (drop multiply and use custom method)
def calculate_efp(result: csr_matrix, probabilities: csr_matrix):
    ni, nl = probabilities.shape
    Efp = np.zeros(nl, dtype=FLOAT_TYPE)
    dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        Efp += result[i].multiply(dense_ones - probabilities[i])

    return Efp


# This is a bit slow, TODO: make it faster (drop multiply and use custom method)
def calculate_efn(result: csr_matrix, probabilities: csr_matrix):
    ni, nl = probabilities.shape
    Efn = np.zeros(nl, dtype=FLOAT_TYPE)
    dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
    for i in range(ni):
        Efn += probabilities[i].multiply(dense_ones - result[i]) 

    return Efn


def csr_macro_population_cm_risk(probabilities: csr_matrix, k: int, measure_func: callable, 
                                tolerance: float = 1e-4, max_iter: int = 10, shuffle_order=True, filename=None, **kwargs):

    # Initialize the prediction variable with some feasible value
    ni, nl = probabilities.shape
    result_data, result_indicies, result_indptr = numba_random_at_k(probabilities.data, probabilities.indices, probabilities.indptr, ni, nl, k)
    
    # For debug set it to first k labels
    #result_data, result_indicies, result_indptr = numba_first_k(probabilities.data, probabilities.indices, probabilities.indptr, ni, nl, k)
    
    result = csr_matrix((result_data, result_indicies, result_indptr), shape=(ni, nl))


    for j in range(max_iter):

        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        # Recalculate expected conf matrices to prevent numerical errors from accumulating too much
        # In this variant they will be all np.matrix with shape (1, nl)
        with Timer():
            Etp = calculate_etp(result, probabilities)

        with Timer():
            Efp = calculate_efp(result, probabilities)
        
        with Timer():
            Efn = calculate_efn(result, probabilities)
        old_score = np.mean(measure_func(Etp / ni, Efp / ni, Efn / ni))

        # Check expected conf matrices
        # print("Etp:", Etp.shape, type(Etp), Etp)
        # print("Efp:", Efp.shape, type(Efp), Efp)
        # print("Efn:", Efn.shape, type(Efn), Efn)

        for i in tqdm(order):
        #for i in order:
            eta = probabilities[i]
            dense_ones = np.ones(nl, dtype=FLOAT_TYPE)
            
            # Adjust local Etp, Efp, Efn
            Etp -= result[i].multiply(eta)
            Efp -= result[i].multiply(dense_ones - eta)
            Efn -= eta.multiply(dense_ones - result[i])

            # Calculate gain and selection
            Etpp = Etp + eta
            Efpp = Efp + (dense_ones - eta)
            Efnn = Efn + eta
            p_score = measure_func(Etpp / ni, Efpp / ni, Efn / ni)
            n_score = measure_func(Etp / ni, Efp / ni, Efnn / ni)
            gains = p_score - n_score
            gains = np.asarray(gains).ravel()
            top_k = np.argpartition(-gains, k)[:k]

            # Update predictions
            result.indices[result.indptr[i]:result.indptr[i+1]] = top_k

            # Update Etp, Efp, Efn
            Etp += result[i].multiply(eta)
            Efp += result[i].multiply(dense_ones - eta)
            Efn += eta.multiply(dense_ones - result[i])

        new_score = np.mean(measure_func(Etp / ni, Efp / ni, Efn / ni))
        print(f"  Iteration {j + 1} finished, expected score: {old_score} -> {new_score}")
        if new_score <= old_score + tolerance:
            break
        
        if filename is not None:
            save_npz(f"{filename}_pred_iter_{j + 1}.npz", result)
            
    return result



@njit
def numba_product_minus_1():
    pass



def csr_block_coordinate_coverage(predictions: np.ndarray, k: int = 5, *, tolerance: float = 1e-4, max_iter: int = 10, shuffle_order=True):
    """
    An efficient implementation of the block coordinate-descent for coverage
    """
    ni, nl = predictions.shape

    # Initialize the prediction variable with some feasible value
    ni, nl = probabilities.shape
    result_data, result_indicies, result_indptr = numba_random_at_k(probabilities.data, probabilities.indices, probabilities.indptr, ni, nl, k)
    
    # For debug set it to first k labels
    #result_data, result_indicies, result_indptr = numba_first_k(probabilities.data, probabilities.indices, probabilities.indptr, ni, nl, k)
    
    result = csr_matrix((result_data, result_indicies, result_indptr), shape=(ni, nl))

    result_x_prediction = result.multiply(predictions)
    #f = np.product(1 - , axis=0)
    old_cov = 1 - np.mean(f)

    predictions = np.minimum(predictions, 1 - 1e-5)

    for j in range(max_iter):
        order = np.arange(ni)
        if shuffle_order:
            np.random.shuffle(order)

        #for i in tqdm(order):
        for i in order:
            # adjust f locally
            f /= 1 - result[i] * predictions[i]

            # calculate gain and selection
            eta = predictions[i, :]
            g = f * eta
            top_k = np.argpartition(-g, k)[:k]

            # Update predictions
            result.indices[result.indptr[i]:result.indptr[i+1]] = top_k

            # update f
            f *= (1 - result[i] * predictions[i])

        new_cov = 1 - np.mean(f)
        # print(f"{old_cov} -> {new_cov}")
        # print(macro_abandonment(predictions, result))
        if new_cov <= old_cov + tolerance:
            break
        old_cov = new_cov

    # print(macro_abandonment(predictions, result))



    return result
