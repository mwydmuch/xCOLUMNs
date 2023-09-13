import time
import json
import numpy as np
from scipy.sparse import csr_matrix


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
