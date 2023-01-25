import numpy as np
from scipy.sparse import csr_matrix
from metrics import macro_precision
from numba import njit

@njit
def sparse_weighted_per_instance(data: np.ndarray, indicies: np.ndarray, indptr: np.ndarray, 
                                weights: np.ndarray, ni: int, nl: int, k: int):
    result_data = np.ones(ni * k, dtype=np.int32)
    result_indicies = np.zeros(ni * k, dtype=np.float32)
    result_indptr = np.zeros(ni + 1, dtype=np.int32)

    # This can be done in parallel, but Numba parallelism seems to not work well here
    for i in range(ni):
        row_data = data[indptr[i]:indptr[i+1]]
        row_indicies = indicies[indptr[i]:indptr[i+1]]
        row_weights = weights[row_indicies].reshape(-1) * row_data
        top_k = row_indicies[np.argsort(-row_weights)[:k]]
        result_indicies[i * k:(i + 1) * k] = top_k
        result_indptr[i + 1] = result_indptr[i] + k

    return result_data, result_indicies, result_indptr

