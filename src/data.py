from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
import os


def load_dataset(path: Path) -> np.ndarray:
    """
    Loads the label matrix from the XMC repository dataset at `path`
    """
    with open(path, "r") as fd:
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


def construct_csr_matrix(data, indices, indptr, dtype=np.float32, sort_indices=False):
    mat = csr_matrix((data, indices, indptr), dtype=dtype)
    if sort_indices:
        mat.sort_indices()
    return mat


def load_txt_labels(path: str):
    """
    Loads the sparse label matrix from the XMC repository dataset
    """
    with open(path, 'r') as file:
        num_ins, num_ftr, num_lbl = file.readline().split(" ")

        data = []
        indices = []
        indptr = [0]
        requires_sort = False

        for line in file:
            labels = line.split(" ")[0].split(",")
            if len(labels) == 1 and labels[0] == "":
                indptr.append(len(indices))
                continue

            prev = -1
            for l in labels:
                ind = int(l)
                indices.append(ind)
                data.append(1.0)
                if prev > ind:
                    requires_sort = True
                prev = ind
            indptr.append(len(indices))

    if requires_sort:
        print("Sorting of the matrix's indices is required. This will take a while ...")

    return construct_csr_matrix(data, indices, indptr, dtype=np.float32, sort_indices=requires_sort)


def load_txt_sparse_pred(path: str):
    """
    Loads the sparse prediction matrix from libsvm like format:
    <label>:<value> <label>:<value> ... 
    """
    with open(path, 'r') as file:
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

    return construct_csr_matrix(data, indices, indptr, dtype=np.float32, sort_indices=requires_sort)


def load_cache_npz_file(path, load_func):
    """
    Loads a npz file from the given path. If the file does not exist, it is created using the given load_func.
    """
    print(f"Loading {path} ...")
    if(not os.path.exists(path + ".npz")):
        print(f"  Creating under {path}.npz for faster loading ...")
        data = load_func(path)
        save_npz(path + ".npz",data)
    else:
        data = load_npz(path + ".npz")

    return data
