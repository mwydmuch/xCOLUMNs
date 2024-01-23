import numpy as np


from xcolumns.block_coordinate import *
from xcolumns.metrics import *
from xcolumns.weighted_prediction import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics_on_conf_matrix import *
from xcolumns.default_types import *
from xcolumns.utils import *


def frank_wolfe_wrapper(
    y_proba,
    utility_func,
    k: int = 5,
    seed: int = 0,
    pred_repeat=1,
    average=False,
    use_last=False,
    y_true_valid=None, y_proba_valid=None,
    **kwargs,
):
    classifiers_a, classifiers_b, classifiers_proba, meta = find_optimal_randomized_classifier_using_frank_wolfe(
        y_true_valid, y_proba_valid, utility_func, max_iters=20, k=k, **kwargs
    )
    print(f"  classifiers weights: {classifiers_proba}")
    y_preds = []
    if use_last:
        print("  using last classifier")
        y_pred = predict_using_randomized_classifier(
            y_proba, classifiers_a[-1:], classifiers_b[-1:], np.array([1]), k=k
        )
        y_preds.append(y_pred)
    elif not average:
        print(f"  predicting with randomized classfier {pred_repeat} times")
        for i in range(pred_repeat):
            y_pred = predict_using_randomized_classifier(
                y_proba, classifiers_a, classifiers_b, classifiers_proba, k=k, seed=seed + i
            )
            y_preds.append(y_pred)
    else:
        print("  averaging classifiers weights")
        avg_classifiers_a = (classifiers_a.multiply(classifiers_proba, axis=1)).sum(axis=0).reshape(1, -1)
        avg_classifiers_b = (classifiers_b.multiply(classifiers_proba, axis=1)).sum(axis=0).reshape(1, -1)
        print(avg_classifiers_a.shape, avg_classifiers_b.shape)
        y_pred = predict_using_randomized_classifier(y_proba, avg_classifiers_a, avg_classifiers_b, np.array([1]), k)
        y_preds.append(y_pred)
    
    meta["thresholds"] = -classifiers_b[-1:] / classifiers_a[-1:]

    return y_preds, meta


def fw_macro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return frank_wolfe_wrapper(
        y_proba, macro_fmeasure_on_conf_matrix, y_true_valid=y_true_valid, y_proba_valid=y_proba_valid, k=k, seed=seed, **kwargs
    )


def fw_macro_f1_on_test(
    y_proba, k: int = 5, seed: int = 0, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return frank_wolfe_wrapper(
        y_proba, macro_fmeasure_on_conf_matrix, y_true_valid=y_true, y_proba_valid=y_proba, k=k, seed=seed, **kwargs
    )


def fw_macro_f1_etu(
    y_proba, k: int = 5, seed: int = 0, y_true_valid=None, y_proba_valid=None, y_true=None, **kwargs
):
    return frank_wolfe_wrapper(
        y_proba, macro_fmeasure_on_conf_matrix, y_true_valid=y_proba, y_proba_valid=y_proba, k=k, seed=seed, **kwargs
    )