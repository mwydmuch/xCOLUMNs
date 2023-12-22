import time
import json
import numpy as np
from scipy.sparse import csr_matrix


def align_dim1(a, b):
    if a.shape[1] != b.shape[1]:
        print("  Fixing shapes ...")
        new_size = max(a.shape[1], b.shape[1])
        a.resize((a.shape[0], new_size))
        b.resize((b.shape[0], new_size))


def loprint(array):
    print(array.shape, array.dtype, array.min(), array.max(), array.mean(), array.std())


class Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self

    def get_time(self):
        return time.time() - self.start

    def __exit__(self, *args):
        print(f"  Time: {self.get_time():>5.2f}s")


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


def lin_search(low, high, step, func):
    best = None
    best_score = None
    for i in np.arange(low, high, step):
        score = func(i)
        if best_score is None or score > best_score:
            best = i
            best_score = score
    return best, best_score


def bin_search(low, high, eps, func):
    while high - low > eps:
        mid = (low + high) / 2
        mid_next = (mid + high) / 2

        if func(mid) < func(mid_next):
            high = mid_next
        else:
            low = mid

    return (low + high) / 2


def ternary_search(low, high, eps, func):
    while high - low > eps:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        if func(mid1) < func(mid2):
            high = mid2
        else:
            low = mid1

    return (low + high) / 2