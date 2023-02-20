import time
import json
import numpy as np
from scipy.sparse import csr_matrix


def loprint(array):
    print(array.shape, array.dtype, array.min(), array.max(), array.mean(), array.std())


def unpack_csr_matrix(matrix: csr_matrix):
    return matrix.data, matrix.indices, matrix.indptr


def unpack_csr_matrices(*matrices):
    result = []
    for m in matrices:
        result.extend(unpack_csr_matrix(m))
    return result


def construct_csr_matrix(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, dtype=None, shape=None, sort_indices=False):
    mat = csr_matrix((data, indices, indptr), dtype=dtype, shape=shape)
    if sort_indices:
        mat.sort_indices()
    return mat


class Timer(object):        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        print (f"  Time: {time.time() - self.start:>5.2f}s")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(filepath, data):
    with open(filepath, "w") as file:
        json.dump(data, file, cls=NpEncoder, indent=4, sort_keys=True)


def load_json(filepath):
    with open(filepath) as file:
        return json.load(file)