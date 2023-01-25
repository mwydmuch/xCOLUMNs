import numpy as np
import time



from metrics import *
from data import *
from prediction import *


test_data = np.array([
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]
])

#print(block_coordinate_descent(test_data, 2))

METRICS = {
    "mC@k": macro_abandonment,
    "iC@k": instance_abandonment, 
    "mP@k": macro_precision,
    "iP@k": instance_precision,
    "mR@k": macro_recall,
    "iR@k": instance_recall,
    "mF@k": macro_f1,
    "iF@k": instance_f1
}

METHODS = {
    #"random": predict_random_at_k,
    "top-k": optimal_instance_precision,
    #"block coord": block_coordinate_ascent_fast,
    # "macro-recall": optimal_macro_recall,
}


class Timer(object):        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        print (f"  Time: {time.time() - self.start:>5.2f}s")


def report_metrics(data, predictions):
    for metric, func in METRICS.items():
        print(f"  {metric}: {100 * func(data, predictions):>5.2f}")


if __name__ == "__main__":
    y_true_path = "datasets/eurlex/eurlex_test.txt"
    eta_pred_path = "predictions/eurlex_top_100"
    # y_true_path = "datasets/amazonCat/amazonCat_test.txt"
    # eta_pred_path = "predictions/amazonCat_top_1000"

    # Create binary files for faster loading
    with Timer():
        y_true = load_cache_npz_file(y_true_path, load_txt_labels)

    with Timer():
        eta_pred = load_cache_npz_file(eta_pred_path, load_txt_sparse_pred)

    # For some spars format this resize might be necessary
    if y_true.shape != eta_pred.shape:
        if y_true.shape[0] != eta_pred.shape[0]:
            raise RuntimeError("Number of instances in true and prediction do not match")
        if y_true.shape[1] != eta_pred.shape[1]:
            eta_pred.resize(y_true.shape)

    # For some testing purposes (with/without should give the same results)
    # y_true = y_true.toarray()
    # eta_pred = eta_pred.toarray()

    # Some old preprocessing code
    # eurlex_pred = np.load("/tmp/predict.npy")
    # eurlex_pred = np.exp(eurlex_pred) / (1 + np.exp(eurlex_pred))
    # eurlex_pred = np.maximum(eurlex_pred - 0.5, 0.5)
    
    for k in (1, 3, 5, 10):
        for method, func in METHODS.items():
            print(f"{method} @ {k}: ")
            with Timer():
                y_pred = func(eta_pred, k)
            report_metrics(y_true, y_pred)

