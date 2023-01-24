import numpy as np
from metrics import *
from data import load_dataset

test_data = np.array([
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0]
])

#print(block_coordinate_descent(test_data, 2))


def report_metrics(data, predictions):
    print(f"  mC@k: {100 * macro_abandonment(data, predictions):>5.2f}  iC@k: {100 * macro_abandonment(data, predictions):.2f}")
    print(f"  mP@k: {100 * macro_precision(data, predictions):>5.2f}  iP@k: {100 * instance_precision(data, predictions):.2f}")
    print(f"  mR@k: {100 * macro_recall(data, predictions):>5.2f}  iR@k: {100 * instance_recall(data, predictions):.2f}")


if False:
    eurlex_data = load_dataset(Path("/mnt/media/datasets/raw/eurlex/test.txt"))
    wiki10_data = load_dataset(Path("/mnt/media/datasets/raw/wiki10/test.txt"))
    amazon_data = load_dataset(Path("/mnt/media/datasets/raw/amazoncat13k/test.txt"))
    test_data = eurlex_data
    for k in (1, 3, 5, 10):
        print("K: ", k)
        print("Random selection: ")
        result = predict_random_at_k(test_data, k)
        report_metrics(test_data, result)
        print("Block coordianate-ascent")
        result = block_coordinate_descent_fast(test_data, k)
        report_metrics(test_data, result)
        print("Optimal Macro-Recall")
        result = optimal_macro_recall(test_data, k)
        report_metrics(test_data, result)
        print()


if True:
    eurlex_data = load_dataset(Path("/mnt/media/datasets/raw/eurlex/test.txt"))
    eurlex_pred = np.load("/tmp/predict.npy")
    eurlex_pred = np.exp(eurlex_pred) / (1 + np.exp(eurlex_pred))
    print(eurlex_pred[0])
    #eurlex_pred = np.maximum(eurlex_pred - 0.5, 0.5)
    for k in (1, 3, 5, 10):
        print("K: ", k)
        print("Top-K: ")
        result = optimal_instance_precision(eurlex_pred, k)
        report_metrics(eurlex_data, result)
        print("Block coordinate-ascent")
        result = block_coordinate_descent_fast(eurlex_pred, k)
        report_metrics(eurlex_data, result)
        print("Optimal Macro-Recall")
        result = optimal_macro_recall(eurlex_pred, k)
        report_metrics(eurlex_data, result)
        print()
