import sys

import click
import numpy as np
from custom_utilities_methods import *
from scipy.sparse import csr_matrix
from tqdm import tqdm
from wrappers_frank_wolfe import *
from wrappers_online_methods import *
from wrappers_threshold_methods import *

from utils import *
from xcolumns.block_coordinate import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics import *
from xcolumns.metrics_original import *
from xcolumns.types import *
from xcolumns.utils import *
from xcolumns.weighted_prediction import *


# TODO: refactor this
RECALCULATE_RESUTLS = False
RECALCULATE_PREDICTION = False

METRICS = {
    "hl": hamming_loss,
    # "min_tp_tn": macro_min_tp_tn,
    # "mC": macro_abandonment,
    # "iC": instance_abandonment,
    "mP": macro_precision,
    "iP": instance_precision,
    "mR": macro_recall,
    "iR": instance_recall,
    "mF": macro_f1,
    "miF": micro_f1_score,
    "iF": instance_f1,
    "mH": macro_hmean,
    "mG": macro_gmean,
    "H": multiclass_hmean,
    "G": multiclass_gmean,
    "Q": multiclass_qmean,
}

METHODS = {
    "default_prediction": (default_prediction, {}),
    "ofo_macro_f1": (ofo_macro, {}),
    "ofo_micro_f1": (ofo_micro, {}),
    # "ofo_etu_macro_f1": (ofo_macro, {"etu_variant": True}),
    # "ofo_etu_micro_f1": (ofo_micro, {"etu_variant": True}),
}

for measure in [
    "macro_f1",
    "micro_f1",
    # "macro_hmean",
    # "macro_gmean",
    # "macro_min_tp_tn",
    # "macro_min_tp_tn_smoothed",
]:
    METHODS.update(
        {
            # f"online_default_{measure}": (eval(f"online_default_{measure}"), {}),
            f"omma_{measure}": (eval(f"omma_{measure}"), {}),
            f"omma_etu_{measure}": (
                eval(f"omma_{measure}"),
                {"etu_variant": True},
            ),
            # f"omma_{measure}_e=2": (eval(f"omma_{measure}"), {"epochs": 2}),
            # f"omma_{measure}_e=3": (eval(f"omma_{measure}"), {"epochs": 3}),
            f"online_greedy_{measure}": (eval(f"online_greedy_{measure}"), {}),
            f"online_greedy_etu_{measure}": (
                eval(f"online_greedy_{measure}"),
                {"etu_variant": True},
            ),
            f"online_frank_wolfe_exp=1.1_{measure}": (
                eval(f"online_frank_wolfe_{measure}"),
                {},
            ),
            f"online_frank_wolfe_etu_exp=1.1_{measure}": (
                eval(f"online_frank_wolfe_{measure}"),
                {"etu_variant": True},
            ),
            # f"online_thresholds_{measure}": (eval(f"online_thresholds_{measure}"), {}),
            # f"find_thresholds_{measure}_on_test": (
            #     eval(f"find_thresholds_{measure}_on_test"),
            #     {},
            # ),
            # f"find_thresholds_{measure}": (eval(f"find_thresholds_{measure}"), {}),
            # f"find_thresholds_{measure}_etu": (
            #     eval(f"find_thresholds_{measure}_etu"),
            #     {},
            # ),
            # f"block_coord_{measure}": (eval(f"bc_{measure}"), {}),
            # f"greedy_{measure}": (
            #     eval(f"bc_{measure}"),
            #     {"init_y_pred": "greedy", "max_iters": 1},
            # ),
            # f"frank_wolfe_{measure}": (eval(f"fw_{measure}"), {}),
            # f"frank_wolfe_on_test_{measure}": (eval(f"fw_{measure}_on_test"), {}),
            # f"frank_wolfe_etu_{measure}": (eval(f"fw_{measure}_etu"), {}),
            # f"frank_wolfe_on_train_etu_{measure}": (
            #     eval(f"fw_{measure}_on_train_etu"),
            #     {},
            # ),
        }
    )


def calculate_and_report_metrics(y_true, y_preds, k, metrics):
    results = {}
    for metric, func in metrics.items():
        values = []
        if not isinstance(y_preds, (list, tuple)):
            y_preds = [y_preds]
        for y_pred in y_preds:
            if y_pred is not None:
                # Check if indeed we have k predictions
                if k > 0:
                    # print(y_pred.sum(axis=1))
                    # print(y_pred[1])
                    # print(y_pred[-1])
                    assert np.all(y_pred.sum(axis=1) == k)
                value = func(y_true, y_pred)
                values.append(value)
        val = val_std = 0
        if len(values) > 0:
            val = np.mean(values)
            val_std = np.std(values)
        results[f"{metric}@{k}"] = val
        results[f"{metric}@{k}_std"] = val_std
        print(f"    {metric}@{k}: {100 * val:>5.2f} +/- {100 * val_std:>5.2f}")

    return results


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("-k", type=int, required=True)
@click.option("-s", "--seed", type=int, required=True)
@click.option("-m", "--method", type=str, required=False, default=None)
@click.option("-p", "--probabilities_path", type=str, required=False, default=None)
@click.option("-l", "--labels_path", type=str, required=False, default=None)
@click.option(
    "-r", "--results_dir", type=str, required=False, default="results_online/"
)
@click.option(
    "--recalculate_predictions", is_flag=True, type=bool, required=False, default=False
)
@click.option(
    "--recalculate_results", is_flag=True, type=bool, required=False, default=False
)
@click.option("--use_dense", is_flag=True, type=bool, required=False, default=False)
def main(
    experiment,
    k,
    seed,
    method,
    probabilities_path,
    labels_path,
    results_dir,
    recalculate_predictions,
    recalculate_results,
    use_dense,
):
    print(experiment)

    if method is None:
        methods = METHODS
    else:
        methods = {k: v for k, v in METHODS.items() if method in k}
        print(methods.keys())
        if len(methods) == 0:
            raise ValueError(f"Unknown method: {method}")

    true_as_pred = "true_as_pred" in experiment
    lightxml_data = "lightxml" in experiment

    plt_loss = "log"

    lightxml_data_load_config = {
        "labels_delimiter": " ",
        "labels_features_delimiter": None,
        "header": False,
    }
    xmlc_data_load_config = {
        "labels_delimiter": ",",
        "labels_features_delimiter": " ",
        "header": True,
    }

    train_y_true_path = None
    train_y_proba_path = None
    valid_y_true_path = None
    valid_y_proba_path = None
    train_y_true = None
    train_y_proba = None
    valid_y_true = None
    valid_y_proba = None

    # Predefined experiments
    if "youtube_deepwalk_online" in experiment:
        y_proba_path = {
            "path": f"datasets/online/youtube_deepwalk/linear_adam_s={seed}/Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/online/youtube_deepwalk/linear_adam_s={seed}/Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "eurlex_lexglue_online" in experiment:
        y_proba_path = {
            "path": f"datasets/online/eurlex_lexglue/linear_adam_s={seed}/Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/online/eurlex_lexglue/linear_adam_s={seed}/Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "mediamill_online" in experiment:
        y_proba_path = {
            "path": f"datasets/online/mediamill/linear_adam_s={seed}/Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/online/mediamill/linear_adam_s={seed}/Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "flicker_deepwalk_online" in experiment:
        y_proba_path = {
            "path": f"datasets/online/flicker_deepwalk/linear_adam_s={seed}/Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/online/flicker_deepwalk/linear_adam_s={seed}/Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "youtube_deepwalk_plt" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/youtube_deepwalk2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/youtube_deepwalk/youtube_deepwalk_test.svm",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/youtube_deepwalk/youtube_deepwalk_train.svm",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/youtube_deepwalk2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "eurlex_lexglue_plt" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/eurlex_lexglue2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/eurlex_lexglue/eurlex_lexglue_test.svm",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/eurlex_lexglue/eurlex_lexglue_train.svm",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/eurlex_lexglue2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "mediamill_plt" in experiment:
        # mediamill - PLT
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/mediamill2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/mediamill/mediamill_test.txt",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/mediamill/mediamill_train.txt",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/mediamill2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "flicker_deepwalk_plt" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/flicker_deepwalk2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/flicker_deepwalk/flicker_deepwalk_test.svm",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/flicker_deepwalk/flicker_deepwalk_train.svm",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/flicker_deepwalk2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "rcv1x_plt" in experiment:
        # RCV1X - PLT + XMLC repo data
        y_proba_path = {
            "path": f"predictions/rcv1x2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/rcv1x/rcv1x_test.txt",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/rcv1x/rcv1x_train.txt",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/rcv1x2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "eurlex_plt" in experiment:
        # Eurlex - PLT + XMLC repo data
        y_proba_path = {
            "path": f"predictions/eurlex2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/eurlex/eurlex_test.txt",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/eurlex/eurlex_train.txt",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/eurlex2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "amazoncat_plt" in experiment:
        # AmazonCat - PLT + XMLC repo data
        y_proba_path = {
            "path": f"predictions/amazonCat2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/amazonCat/amazonCat_test.txt",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/amazonCat/amazonCat_train.txt",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/amazonCat2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "wiki10_plt" in experiment:
        # Wiki10 - PLT + XMLC repo data
        y_proba_path = {
            "path": f"predictions/wiki102_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/wiki10/wiki10_test.txt",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/wiki10/wiki10_train.txt",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/wiki102_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "sensorless_hsm" in experiment:
        # Wiki10 - PLT + XMLC repo data
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/sensorless_top_12_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/sensorless/sensorless_test.txt",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/sensorless/sensorless_train.txt",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/sensorless_train_top_12_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "news20_hsm" in experiment:
        # Wiki10 - PLT + XMLC repo data
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/news20_top_20_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/news20/news20.test",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/news20/news20.train",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/news20_train_top_20_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "cal101_hsm" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/cal1012_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/FLAT_CAL101/FLAT_CAL101.test",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/FLAT_CAL101/FLAT_CAL101.train",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/cal1012_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "cal256_hsm" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/cal2562_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/FLAT_CAL256/FLAT_CAL256.test",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/FLAT_CAL256/FLAT_CAL256.train",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/cal2562_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "protein_hsm" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/protein2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/FLAT_PROTEIN2/FLAT_PROTEIN2.test",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/FLAT_PROTEIN2/FLAT_PROTEIN2.train",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/protein2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "aloi.bin_hsm" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/aloi.bin2_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/aloi.bin/aloi.bin.test",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/aloi.bin/aloi.bin.train",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/aloi.bin2_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "ledgar_hsm" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/ledgar_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/ledgar/ledgar_test.svm",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/ledgar/ledgar_train.svm",
            "load_func": load_txt_labels,
        }
        train_y_proba_path = {
            "path": f"predictions/ledgar_train_top_100_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }

    elif "sensorless_lr" in experiment:
        y_proba_path = {
            "path": f"datasets/sklearn/sensorless_lr/test_Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/sklearn/sensorless_lr/test_Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "news20_lr" in experiment:
        y_proba_path = {
            "path": f"datasets/sklearn/news20_lr/test_Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/sklearn/news20_lr/test_Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "ledgar_lr" in experiment:
        y_proba_path = {
            "path": f"datasets/sklearn/ledgar_lr/test_Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/sklearn/ledgar_lr/test_Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "cal101_lr" in experiment:
        y_proba_path = {
            "path": f"datasets/sklearn/cal101_lr/test_Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/sklearn/cal101_lr/test_Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "cal256_lr" in experiment:
        y_proba_path = {
            "path": f"datasets/sklearn/cal256_lr/test_Y_pred.npz",
            "load_func": load_npz_wrapper,
        }
        y_true_path = {
            "path": f"datasets/sklearn/cal256_lr/test_Y_true.npz",
            "load_func": load_npz_wrapper,
        }

    elif "eurlex_lightxml" in experiment:
        # Eurlex - LightXML
        y_true_path = {
            "path": "datasets/EUR-Lex/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/eurlex/eurlex4k_full_plain-scores.npy",
            "load_func": load_npy_full_pred,
            "keep_top_k": 100,
            "apply_sigmoid": True,
        }
        train_y_true_path = {
            "path": "datasets/EUR-Lex/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "amazoncat_lightxml" in experiment:
        # Wiki - LightXML
        y_true_path = {
            "path": "datasets/AmazonCat-13K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/amazonCat_top_100.notnpz",
            "load_func": load_npz_wrapper,
            "apply_sigmoid": True,
        }
        train_y_true_path = {
            "path": "datasets/AmazonCat-13K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "wiki10_lightxml" in experiment:
        # Wiki - LightXML
        y_true_path = {
            "path": "datasets/Wiki10-31K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/wiki10_top_100.notnpz",
            "load_func": load_npz_wrapper,
            "apply_sigmoid": True,
        }
        train_y_true_path = {
            "path": "datasets/Wiki10-31K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "amazon_lightxml" in experiment:
        # Amazon - LightXML
        y_true_path = {
            "path": "datasets/Amazon-670K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/amazon/amazon670k_light_t0",
            "load_func": load_npy_sparse_pred,
        }
        train_y_true_path = {
            "path": "datasets/Amazon-670K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "amazon_1000_lightxml" in experiment:
        # Amazon - LightXML
        y_true_path = {
            "path": "datasets/Amazon-670K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/amazon_1000/amazon670k_light_original_t0",
            "load_func": load_npy_sparse_pred,
        }
        train_y_true_path = {
            "path": "datasets/Amazon-670K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "wiki500_lightxml" in experiment:
        # WikiLarge - LightXML
        y_true_path = {
            "path": "datasets/Wiki-500K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/wiki500/wiki500k_light_t0",
            "load_func": load_npy_sparse_pred,
        }
        train_y_true_path = {
            "path": "datasets/Wiki-500K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "wiki500_1000_lightxml" in experiment:
        # WikiLarge - LightXML
        y_true_path = {
            "path": "datasets/Wiki-500K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/wiki500_1000/wiki500k_light_original_t0",
            "load_func": load_npy_sparse_pred,
        }
        train_y_true_path = {
            "path": "datasets/Wiki-500K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    else:
        raise RuntimeError(f"No matching experiment: {experiment}")

    # Remap labels for LightXML predictions and use it when loading data
    if lightxml_data:
        with Timer():
            labels_map = calculate_lightxml_labels(
                train_y_true_path["path"], y_true_path["path"]
            )
        train_y_true_path.update(lightxml_data_load_config)
        y_true_path.update(lightxml_data_load_config)
        train_y_true_path["labels_map"] = labels_map
        y_true_path["labels_map"] = labels_map
    else:
        if train_y_true_path is not None:
            train_y_true_path.update(xmlc_data_load_config)
        y_true_path.update(xmlc_data_load_config)

    # Create binary files for faster loading
    with Timer():
        y_true = load_cache_npz_file(**y_true_path)

    with Timer():
        y_proba = load_cache_npz_file(**y_proba_path)

    if train_y_true_path is not None:
        with Timer():
            train_y_true = load_cache_npz_file(**train_y_true_path)

    if train_y_proba_path is not None:
        with Timer():
            train_y_proba = load_cache_npz_file(**train_y_proba_path)

    if valid_y_true_path is not None:
        with Timer():
            valid_y_true = load_cache_npz_file(**valid_y_true_path)

    if valid_y_proba_path is not None:
        with Timer():
            valid_y_proba = load_cache_npz_file(**valid_y_proba_path)

    # For some sparse format this resize might be necessary
    if y_true.shape != y_proba.shape:
        print(y_true.shape, type(y_true), type(y_proba))
        print(y_true, len(y_true.indices), len(y_true.data.astype(np.int32)))
        new_indices = np.zeros(y_proba.shape[0], dtype=np.int32)
        new_indices[y_true.indices] = y_true.data.astype(np.int32)
        if y_true.shape[1] == y_proba.shape[0]:
            y_true = csr_matrix(
                (
                    np.ones(y_proba.shape[0]),
                    new_indices,
                    np.arange(y_true.shape[1] + 1),
                ),
                shape=(y_proba.shape[0], y_proba.shape[1]),
                dtype=np.float32,
            )

        if y_true.shape[0] != y_proba.shape[0]:
            raise RuntimeError(
                f"Number of instances in true and prediction do not match {y_true.shape[0]} != {y_proba.shape[0]}"
            )
        align_dim1(y_true, y_proba)

    # Calculate priors and propensities
    priors = None
    inv_ps = None
    with Timer():
        if train_y_true_path is not None:
            print("Calculating priors and propensities")
            priors = labels_priors(train_y_true)
            # priors = labels_priors(y_true)
            inv_ps = jpv_inverse_propensity(train_y_true)

    if "apply_sigmoid" in y_proba_path and y_proba_path["apply_sigmoid"]:
        # LightXML predictions aren't probabilities
        y_proba.data = 1.0 / (1.0 + np.exp(-y_proba.data))

    # Use true labels as predictions with 1.0 score (probability)
    if true_as_pred:
        y_proba = y_true

    print(f"y_true: type={type(y_true)}, shape={y_true.shape}")
    print(f"y_proba: type={type(y_proba)}, shape={y_proba.shape}")
    if train_y_true is not None:
        print(f"train_y_true: type={type(train_y_true)}, shape={train_y_true.shape}")
    if train_y_proba is not None:
        print(f"train_y_proba: type={type(train_y_proba)}, shape={train_y_proba.shape}")

    # Convert to array to check if it gives the same results
    if use_dense:
        y_true = y_true.toarray()
        y_proba = y_proba.toarray()
        train_y_true = train_y_true.toarray()
        if train_y_proba is not None:
            train_y_proba = train_y_proba.toarray()
        # train_y_true = train_y_true.toarray()
        print(y_proba.shape, y_true.shape, y_proba.dtype, y_true.dtype)

    output_path_prefix = f"{results_dir}/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for method, func in methods.items():
        print(f"{experiment} - {method} @ {k} (seed {seed}): ")

        output_path = f"{output_path_prefix}{method}_k={k}_s={seed}"
        results_path = f"{output_path}_results.json"
        pred_path = f"{output_path}_pred.npz"

        func[1]["return_meta"] = True  # Include meta data in result
        func[1]["y_true"] = y_true
        func[1]["y_true_valid"] = train_y_true

        if "online" in experiment:
            func[1]["shuffle_order"] = False

        if train_y_proba is not None:
            func[1]["y_proba_valid"] = train_y_proba

        if (
            not os.path.exists(results_path)
            or recalculate_results
            or recalculate_predictions
        ):
            results = {}
            if not os.path.exists(pred_path) or recalculate_predictions:
                try:
                    y_pred, meta = func[0](
                        y_proba,
                        k,
                        priors=priors,
                        inv_ps=inv_ps,
                        seed=seed,
                        **func[1],
                    )
                except Exception as e:
                    print(f"  Error: {e}")
                    continue

                # print(y_proba[-1], y_proba_copy[-1])
                print("  Evaluating predictions")
                results.update(meta)
                # print(f"  Iters: {meta['iters']}")
                print(f"  Time: {meta['time']:>5.2f} s")
                # save_npz_wrapper(pred_path, y_pred)
                save_json(results_path, results)
            else:
                # y_pred = load_npz_wrapper(pred_path)
                results = load_json(results_path)

            print("  Metrics (%):")
            results.update(calculate_and_report_metrics(y_true, y_pred, k, METRICS))
            save_json(results_path, results)

        print("  Done")


if __name__ == "__main__":
    main()
