import numpy as np
from scipy.sparse import csr_matrix
import torch
import pickle
import os
import psutil


def load_obj(file_name):
    with open(file_name, "rb") as file:
        return pickle.load(file)


def save_obj(file_name, obj):
    with open(file_name, "wb") as file:
        pickle.dump(obj, file)


def _get_dtype(first_element):
    if isinstance(first_element, (list, tuple)):
        return type(first_element[1])
    else:
        return type(first_element)


def _get_first_element_of_list_of_lists(X):
    for x in X:
        if len(x):
            return x[0]
    raise ValueError("X is does not contain any element")


def to_csr_matrix(X, shape=None, sort_indices=False, dtype=np.float32):
    """
    Converts sparse matrix-like data, like list of list of tuples (idx, value), to Scipy csr_matrix.

    :param X: Matrix-like object to convert to csr_matrix: ndarray or list of lists of ints or tuples of ints and floats (idx, value).
    :type X: ndarray, list[list[int|str]], list[list[tuple[int, float]]
    :param shape: Shape of the matrix, if None, shape will be deduce from X, defaults to None
    :type shape: tuple, optional
    :param sort_indices: Sort rows' data by indices (idx), defaults to False
    :type sort_indices: bool, optional
    :param dtype: Data type of the matrix, defaults to np.float32
    :type dtype: type, optional
    :return: X as csr_matrix.
    :rtype: csr_matrix
    """
    if isinstance(X, list) and isinstance(X[0], (list, tuple, set)):
        first_element = _get_first_element_of_list_of_lists(X)

        if dtype is None:
            dtype = _get_dtype(first_element)

        size = 0
        for x in X:
            size += len(x)

        indptr = np.zeros(len(X) + 1, dtype=np.int32)
        indices = np.zeros(size, dtype=np.int32)
        data = np.ones(size, dtype=dtype)
        cells = 0

        if isinstance(first_element, tuple):
            for row, x in enumerate(X):
                indptr[row] = cells
                if sort_indices:
                    x = sorted(x)
                for x_i in x:
                    indices[cells] = x_i[0]
                    data[cells] = x_i[1]
                    cells += 1
            indptr[len(X)] = cells

        else:
            for row, x in enumerate(X):
                indptr[row] = cells
                indices[cells : cells + len(x)] = sorted(x) if sort_indices else x
                cells += len(x)
            indptr[len(X)] = cells

        array = csr_matrix((data, indices, indptr), shape=shape)
    else:  # Try to convert via constructor
        array = csr_matrix(X, dtype=dtype, shape=shape)
    # raise TypeError('Cannot convert X to csr_matrix')

    # Check type
    if array.dtype != dtype:
        print("Conversion", array.dtype, dtype)
        array = array.astype(dtype)

    return array


def csr_vec_to_sparse_tensor(csr_vec):
    i = torch.LongTensor([list(csr_vec.indices)])
    v = torch.FloatTensor(csr_vec.data)
    tensor = torch.sparse.FloatTensor(i, v, torch.Size([csr_vec.shape[1]]))
    return tensor


def csr_vec_to_dense_tensor(csr_vec):
    tensor = torch.zeros(csr_vec.shape[1], dtype=torch.float)
    tensor[csr_vec.indices] = torch.tensor(csr_vec.data)
    return tensor


def tp_at_k(output, target, top_k):
    top_k_idx = torch.argsort(output, dim=1, descending=True)[:, :top_k]
    return target[top_k_idx].sum(dim=1)


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)
