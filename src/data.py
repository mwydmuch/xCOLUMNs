from pathlib import Path
import numpy as np


def load_dataset(path: Path) -> np.ndarray:
    """
    Loads the label matrix from the XMC repository dataset at `path`
    """
    with path.open() as fd:
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
