import json
import os
import pickle
import time
from math import log, log2
from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm

from xcolumns.utils import construct_csr_matrix


def call_function_with_supported_kwargs(func, *args, **kwargs):
    if hasattr(func, "__signature__"):
        params = func.__signature__.parameters
    else:
        params = func.__code__.co_varnames
    if "kwargs" not in params:
        selected_kwargs = {k: v for k, v in kwargs.items() if k in params}
    else:
        selected_kwargs = kwargs
    # print(f"  Calling {func.__name__} with kwargs: {selected_kwargs=}")
    return func(*args, **selected_kwargs)


def align_dim1(a, b):
    if a.shape[1] != b.shape[1]:
        print("  Fixing shapes ...")
        new_size = max(a.shape[1], b.shape[1])
        a.resize((a.shape[0], new_size))
        b.resize((b.shape[0], new_size))


def align_dim1_new(a, b):
    if a.shape[1] != b.shape[1]:
        print("  Fixing shapes ...")
        new_size = max(a.shape[1], b.shape[1])
        a = np.resize(a, (a.shape[0], new_size))
        b = np.resize(b, (b.shape[0], new_size))
    return a, b


def loprint(array):
    print(array.shape, array.dtype, array.min(), array.max(), array.mean(), array.std())


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


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def get_time(self):
        return time.time() - self.start

    def __exit__(self, *args):
        print(f"  Time: {self.get_time():>5.2f}s")


def load_dataset(path: Path) -> np.ndarray:
    """
    Loads the label matrix from the XMC repository dataset at `path`
    """
    with open(path) as fd:
        num_ins, num_ftr, num_lbl = fd.readline().split(" ")

        label_matrix = np.zeros((int(num_ins), int(num_lbl)), dtype=np.float32)

        for i, line in enumerate(fd):
            data = line.split(" ")
            # first entry is the labels
            split = data[0].split(",")
            if len(split) == 1 and split[0] == "":
                continue
            else:
                lbls = np.fromiter(map(int, split), dtype=np.int32)
                label_matrix[i, lbls] = 1
        return label_matrix


def load_npy_dataset(path: str, num_ins, num_lbl):
    data = np.load(path, allow_pickle=True)
    label_matrix = np.zeros((int(num_ins), int(num_lbl)), dtype=np.float32)
    for i, lbl_list in enumerate(data):
        label_matrix[i, lbl_list] = 1
    return label_matrix


def load_txt_labels(
    path: str,
    header=True,
    labels_delimiter=",",
    labels_features_delimiter=" ",
    labels_map: dict = None,
):
    """
    Loads the sparse label matrix from the XMC repository dataset or similar text format
    """
    with open(path) as file:
        data = []
        indices = []
        indptr = [0]
        requires_sort = False

        if header:
            num_ins, num_ftr, num_lbl = file.readline().split(" ")

        for i, line in enumerate(file):
            labels = line
            if labels_features_delimiter is not None:
                labels = line.split(labels_features_delimiter)[0]
            labels = labels.split(labels_delimiter)
            if len(labels) == 1 and labels[0] == "":
                indptr.append(len(indices))
                continue

            prev = -1
            for l in labels:
                if labels_map is not None:
                    ind = labels_map[l.strip()]
                else:
                    ind = int(l)
                indices.append(ind)
                data.append(1.0)
                if prev > ind:
                    requires_sort = True
                prev = ind
            indptr.append(len(indices))

    if requires_sort:
        print(
            "  Sorting of the matrix's indices is required. This may take a while ..."
        )

    return construct_csr_matrix(
        data, indices, indptr, dtype=np.float32, sort_indices=requires_sort
    )


def load_txt_sparse_pred(path: str):
    """
    Loads the sparse prediction matrix from libsvm like format:
    <label>:<value> <label>:<value> ...
    """
    with open(path) as file:
        data = []
        indices = []
        indptr = [0]
        requires_sort = False

        for line in file:
            prev = -1
            for p in line.split():
                ind, val = p.split(":")
                ind = int(ind)
                indices.append(ind)
                data.append(float(val))
                if prev > ind:
                    requires_sort = True
                prev = ind
            indptr.append(len(indices))

    return construct_csr_matrix(
        data, indices, indptr, dtype=np.float32, sort_indices=requires_sort
    )


def load_npy_sparse_pred(path: str):
    indices = np.load(path + "-labels.npy", allow_pickle=True)
    data = np.load(path + "-scores.npy", allow_pickle=True)
    indptr = np.arange(0, indices.shape[0] + 1, 1, dtype=np.int32) * indices.shape[1]
    return construct_csr_matrix(
        data.flatten(), indices.flatten(), indptr, dtype=np.float32, sort_indices=True
    )


def load_npy_full_pred(path: str, keep_top_k: int = 0, **kwargs):
    dense_data = np.load(path, allow_pickle=True)

    if keep_top_k < 0:
        k = dense_data.shape[1] - k - 1

    data = -np.partition(-dense_data, keep_top_k, axis=1)[:, :keep_top_k]
    indices = np.argpartition(-dense_data, keep_top_k, axis=1)[:, :keep_top_k]
    indptr = np.arange(0, indices.shape[0] + 1, 1, dtype=np.int32) * indices.shape[1]
    return construct_csr_matrix(
        data.flatten(), indices.flatten(), indptr, dtype=np.float32, sort_indices=True
    )


def load_cache_npz_file(
    path: Union[str, Path], load_func: callable, recreate=False, **load_func_args
):
    """
    Loads a npz file from the given path. If the file does not exist, it is created using the given load_func.
    """
    print(f"Loading {path} ...")
    if not os.path.exists(path + ".npz") or recreate:
        print(f"  Creating cache under {path}.npz for faster loading ...")
        data = load_func(path, **load_func_args)
        save_npz(path + ".npz", data)
    else:
        data = load_npz(path + ".npz")

    return data


def count_labels(Y: Union[np.ndarray, csr_matrix]):
    """
    Count number of occurrences of each label.
    """
    counts = np.ravel(Y.sum(axis=0))

    return counts


def labels_priors(Y: Union[np.ndarray, csr_matrix]):
    """
    Compute the prior probability of each label.
    """
    counts = count_labels(Y)
    priors = counts / Y.shape[0]

    return priors


def jpv_propensity(Y: Union[np.ndarray, csr_matrix], A=0.55, B=1.5):
    """
    Compute the propensity of each label.
    """
    return 1.0 / jpv_inverse_propensity(Y, A, B)


def jpv_inverse_propensity(Y: Union[np.ndarray, csr_matrix], A=0.55, B=1.5):
    """
    Compute the inverse propensity of each label.
    """
    counts = count_labels(Y)
    n = Y.shape[0]
    C = (log(n) - 1) * (B + 1) ** A
    inv_ps = 1 + C * (counts + B) ** -A
    return inv_ps


def load_weights_file(filepath):
    with open(filepath) as file:
        v = []
        for line in file:
            v.append(float(line.strip()))
        return v


def calculate_lightxml_labels(train_data_path, test_data_path):
    print("Creating lightxml label map ...")
    label_map = {}

    with open(train_data_path) as f:
        for i in tqdm(f):
            for l in i.replace("\n", "").split():
                label_map[l.strip()] = 0

    with open(test_data_path) as f:
        for i in tqdm(f):
            for l in i.replace("\n", "").split():
                label_map[l.strip()] = 0

    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k] = i

    return label_map


def load_npz_wrapper(path: Union[str, Path], **kwargs):
    return load_npz(path)


def save_npz_wrapper(path: Union[str, Path], data: csr_matrix, **kwargs):
    save_npz(path, data)


# Store data (serialize)
def save_pickle(path, data, **kwargs):
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Load data (deserialize)
def load_pickle(path, **kwargs):
    with open(path, "rb") as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data
