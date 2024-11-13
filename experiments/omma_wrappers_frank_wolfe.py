import numpy as np
from custom_utilities_methods import *

from utils import *
from xcolumns.block_coordinate import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics import *
from xcolumns.types import *
from xcolumns.utils import *
from xcolumns.weighted_prediction import *


def frank_wolfe_wrapper(
    y_proba,
    utility_func,
    k: int = 5,
    seed: int = 0,
    pred_repeat=1,
    average=False,
    use_last=False,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if y_true_valid is None:
        return no_support(y_proba)

    rnd_cls, meta = call_function_with_supported_kwargs(
        find_classifier_using_fw,
        y_true_valid,
        y_proba_valid,
        utility_func,
        max_iters=50,
        k=k,
        seed=seed,
        **kwargs,
    )
    classifiers_a = rnd_cls.a
    classifiers_b = rnd_cls.b
    classifiers_proba = rnd_cls.p
    print(f"  classifiers weights: {classifiers_proba}")
    y_preds = []
    if use_last:
        print("  using last classifier")
        y_pred = predict_using_randomized_weighted_classifier(
            y_proba, k, classifiers_a[-1:], classifiers_b[-1:], np.array([1])
        )
        y_preds.append(y_pred)
    elif not average:
        print(f"  predicting with randomized classfier {pred_repeat} times")
        for i in range(pred_repeat):
            y_pred = predict_using_randomized_weighted_classifier(
                y_proba,
                k,
                classifiers_a,
                classifiers_b,
                classifiers_proba,
                seed=seed + i,
            )
            y_preds.append(y_pred)
    else:
        print("  averaging classifiers weights")
        avg_classifiers_a = np.zeros_like(classifiers_a[0])
        avg_classifiers_b = np.zeros_like(classifiers_b[0])
        for i in range(len(classifiers_proba)):
            avg_classifiers_a += classifiers_a[i] * classifiers_proba[i]
            avg_classifiers_b += classifiers_b[i] * classifiers_proba[i]
        avg_classifiers_a = avg_classifiers_a.reshape(1, -1)
        avg_classifiers_b = avg_classifiers_b.reshape(1, -1)
        avg_classifiers_a /= classifiers_proba.sum()
        avg_classifiers_b /= classifiers_proba.sum()
        y_pred = predict_using_randomized_weighted_classifier(
            y_proba, k, avg_classifiers_a, avg_classifiers_b, np.array([1])
        )
        y_preds.append(y_pred)

    if y_true is not None:
        utils = []
        for y_pred in y_preds:
            C = calculate_confusion_matrix(y_true, y_pred)
            utility_func_value = utility_func(*C)
            utils.append(utility_func_value)

        meta["pred_utility_history"] = [(y_true.shape[0], np.mean(utils))]
    print(f"  utility={meta['pred_utility_history']}")

    return y_preds, meta


def fw_macro_f1(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_f1_score_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_f1_on_train_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_f1_score_on_conf_matrix,
        y_true_valid=y_proba_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_f1_on_test(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_f1_score_on_conf_matrix,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_f1_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_f1_score_on_conf_matrix,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_micro_f1(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        micro_f1_score_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_micro_f1_on_train_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        micro_f1_score_on_conf_matrix,
        y_true_valid=y_proba_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_micro_f1_on_test(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        micro_f1_score_on_conf_matrix,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_micro_f1_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        micro_f1_score_on_conf_matrix,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_precision(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if k == 0:
        raise ValueError("k must be > 0")

    return frank_wolfe_wrapper(
        y_proba,
        macro_precision_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_precision_on_train_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if k == 0:
        raise ValueError("k must be > 0")

    return frank_wolfe_wrapper(
        y_proba,
        macro_precision_on_conf_matrix,
        y_true_valid=y_proba_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_precision_on_test(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if k == 0:
        raise ValueError("k must be > 0")

    return frank_wolfe_wrapper(
        y_proba,
        macro_precision_on_conf_matrix,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_precision_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if k == 0:
        raise ValueError("k must be > 0")

    return frank_wolfe_wrapper(
        y_proba,
        macro_precision_on_conf_matrix,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_recall(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if k == 0:
        raise ValueError("k must be > 0")

    return frank_wolfe_wrapper(
        y_proba,
        macro_recall_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_recall_on_train_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if k == 0:
        raise ValueError("k must be > 0")

    return frank_wolfe_wrapper(
        y_proba,
        macro_recall_on_conf_matrix,
        y_true_valid=y_proba_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_recall_on_test(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if k == 0:
        raise ValueError("k must be > 0")

    return frank_wolfe_wrapper(
        y_proba,
        macro_recall_on_conf_matrix,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_recall_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    if k == 0:
        raise ValueError("k must be > 0")

    return frank_wolfe_wrapper(
        y_proba,
        macro_recall_on_conf_matrix,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_min_tp_tn(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_min_tp_tn_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_min_tp_tn_on_test(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_min_tp_tn_on_conf_matrix,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_min_tp_tn_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_min_tp_tn_on_conf_matrix,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_hmean(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_hmean_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_hmean_on_train_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_hmean_on_conf_matrix,
        y_true_valid=y_proba_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_hmean_on_test(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_hmean_on_conf_matrix,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_hmean_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_hmean_on_conf_matrix,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_gmean(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_gmean_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_gmean_on_train_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_gmean_on_conf_matrix,
        y_true_valid=y_proba_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_gmean_on_test(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_gmean_on_conf_matrix,
        y_true_valid=y_true,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_macro_gmean_etu(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        macro_gmean_on_conf_matrix,
        y_true_valid=y_proba,
        y_proba_valid=y_proba,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_multiclass_gmean(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        log_multiclass_gmean_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_multiclass_qmean(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        multiclass_qmean_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )


def fw_multiclass_hmean(
    y_proba,
    k: int = 5,
    seed: int = 0,
    y_true_valid=None,
    y_proba_valid=None,
    y_true=None,
    **kwargs,
):
    return frank_wolfe_wrapper(
        y_proba,
        multiclass_hmean_on_conf_matrix,
        y_true_valid=y_true_valid,
        y_proba_valid=y_proba_valid,
        k=k,
        seed=seed,
        y_true=y_true,
        **kwargs,
    )
