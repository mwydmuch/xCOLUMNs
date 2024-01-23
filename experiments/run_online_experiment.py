import sys

import click
import numpy as np
from utils import *
from tqdm import tqdm

from xcolumns.block_coordinate import *
from xcolumns.metrics import *
from xcolumns.weighted_prediction import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics_on_conf_matrix import *
from xcolumns.default_types import *
from xcolumns.utils import *

from wrappers_frank_wolfe import *
from wrappers_online_methods import *
from wrappers_threshold_methods import *
from custom_utilities_methods import *


# TODO: refactor this
RECALCULATE_RESUTLS = False
RECALCULATE_PREDICTION = False

METRICS = {
    "mC": macro_abandonment,
    "iC": instance_abandonment,
    "mP": macro_precision,
    "iP": instance_precision,
    "mR": macro_recall,
    "iR": instance_recall,
    "mF": macro_f1,
    "iF": instance_f1,
}



METHODS = {
    # "online-block-coord-macro-f1": (online_bc_macro_f1, {}),
    # "online-greedy-block-coord-macro-f1": (online_bc_macro_f1, {"greedy": True, "num_valid_sets": 1, "valid_set_size": 1}),
    # "single-online-block-coord-macro-f1": (online_bc_macro_f1, {"num_valid_sets": 1, "valid_set_size": 1}),
    # #"single-online-block-coord-on-true-macro-f1": (online_bc_macro_f1, {"num_valid_sets": 1, "valid_set_size": 1, "use_true_valid_bc": True}),
    # "single-online-block-coord-with-true-c-macro-f1": (online_bc_macro_f1, {"num_valid_sets": 1, "valid_set_size": 1, "use_true_valid_c": True}),
    # "online-block-coord-with-fw-macro-f1": (online_bc_with_fw_macro_f1, {}),
    # "online-greedy-block-coord-with-fw-macro-f1": (online_bc_with_fw_macro_f1, {"greedy": True, "num_valid_sets": 1, "valid_set_size": 1}),
    # "single-online-block-coord-with-fw-macro-f1": (online_bc_with_fw_macro_f1, {"num_valid_sets": 1, "valid_set_size": 1}),
# }
# METHODS2={
    # "frank-wolfe-macro-f1": (fw_macro_f1, {}),
    # "frank-wolfe-average-macro-f1": (fw_macro_f1, {"average": True}),
    # "frank-wolfe-last-macro-f1": (fw_macro_f1, {"use_last": True}),

    #"optimal-instance-prec": (predict_for_optimal_instance_precision, {}),
    
    # #"frank-wolfe-macro-f1-averaged-classifier-on-test": (fw_macro_f1_on_test, {"average": True}),
    # "frank-wolfe-macro-f1-last-classifier": (fw_macro_f1, {"use_last": True}),

    # #"frank-wolfe-macro-f1-averaged-classifier-on-test": (fw_macro_f1_on_test, {"average": True}),
    # "frank-wolfe-macro-f1-last-classifier-on-test": (fw_macro_f1_on_test, {"use_last": True}),
    
    "default-prediction": (default_prediction, {}),

    "online-gd-macro-f1": (online_gd_macro_f1, {}),
    "online-greedy-macro-f1": (online_greedy_macro_f1, {}),
    "online-frank-wolfe-macro-f1": (online_fw_macro_f1, {}),

    "online-gd-macro-f1-x2": (online_gd_macro_f1, {"epochs": 2}),
    #"online-gd-macro-f1-x5": (online_gd_macro_f1, {"epochs": 5}),
    #"online-gd-macro-f1-x10": (online_gd_macro_f1, {"epochs": 10}),
    #"online-frank-wolfe-macro-f1-x5": (online_fw_macro_f1, {"epochs": 5}),

    # "frank-wolfe-macro-f1": (fw_macro_f1, {}),
    # "frank-wolfe-macro-f1-on-test": (fw_macro_f1_on_test, {}),
    # "frank-wolfe-macro-f1-etu": (fw_macro_f1_etu, {}),
    
    # "block-coord-macro-f1": (bc_macro_f1, {}),
    # "greedy-macro-f1": (bc_macro_f1, {"init_y_pred": "greedy", "max_iter": 1}),
    
    # "find-thresholds-macro-f1-on-test": (find_thresholds_macro_f1_on_test, {}),
    # "find-thresholds-macro-f1": (find_thresholds_macro_f1, {}),
    # "find-thresholds-macro-f1-etu": (find_thresholds_macro_f1_etu, {}),


    #"online-thresholds-macro-f1": (online_thresholds_macro_f1, {}),

    # min tp-tn
    # "block-coord-macro-min-tp-tn": (bc_macro_min_tp_tn, {}),
    # "greedy-macro-min-tp-tn": (bc_macro_min_tp_tn, {"init_y_pred": "greedy", "max_iter": 1}),
    # "online-greedy-macro-min-tp-tn": (online_greedy_macro_min_tp_tn, {}),

    # "find-thresholds-macro-min-tp-tn-on-test": (find_thresholds_macro_min_tp_tn_on_test, {}),
    # "find-thresholds-macro-min-tp-tn": (find_thresholds_macro_min_tp_tn, {}),
    # "find-thresholds-macro-min-tp-tn-etu": (find_thresholds_macro_min_tp_tn_etu, {}),
}



def calculate_and_report_metrics(y_true, y_preds, k, metrics):
    results = {}
    for metric, func in metrics.items():
        values = []
        if not isinstance(y_preds, (list, tuple)):
            y_preds = [y_preds]
        for y_pred in y_preds:
            # Check if indeed we have k predictions
            if k > 0:
                assert np.all(y_pred.sum(axis=1) == k)
            value = func(y_true, y_pred)
            values.append(value)
        results[f"{metric}@{k}"] = np.mean(values)
        results[f"{metric}@{k}_std"] = np.std(values)
        print(
            f"    {metric}@{k}: {100 * np.mean(values):>5.2f} +/- {100 * np.std(values):>5.2f}"
        )

    return results


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("-k", type=int, required=True)
@click.option("-s", "--seed", type=int, required=True)
@click.option("-m", "--method", type=str, required=False, default=None)
@click.option("-p", "--probabilities_path", type=str, required=False, default=None)
@click.option("-l", "--labels_path", type=str, required=False, default=None)
@click.option("-r", "--results_dir", type=str, required=False, default="results_bca4")
def main(experiment, k, seed, method, probabilities_path, labels_path, results_dir):
    print(experiment)

    if method is None:
        methods = METHODS
    elif method in METHODS:
        methods = {method: METHODS[method]}
    else:
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
    if "youtube_deepwalk_plt" in experiment:
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
        if y_true.shape[0] != y_proba.shape[0]:
            raise RuntimeError(
                f"Number of instances in true and prediction do not match {y_true.shape[0]} != {y_proba.shape[0]}"
            )
        align_dim1(y_true, y_proba)

    # Calculate priors and propensities
    with Timer():
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
    y_true = y_true.toarray()
    y_proba = y_proba.toarray()
    train_y_true = train_y_true.toarray()
    train_y_proba = train_y_proba.toarray()
    y_proba_copy = y_proba.copy()
    # train_y_true = train_y_true.toarray()

    output_path_prefix = f"results_online2/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for method, func in methods.items():
        print(f"{experiment} - {method} @ {k}: ")

        output_path = f"{output_path_prefix}{method}_k={k}_s={seed}"
        results_path = f"{output_path}_results.json"
        pred_path = f"{output_path}_pred.npz"

        func[1]["return_meta"] = True  # Include meta data in result
        func[1]["y_true"] = y_true
        func[1]["y_true_valid"] = train_y_true
        func[1]["y_proba_valid"] = train_y_proba

        if not os.path.exists(results_path) or RECALCULATE_RESUTLS:
            results = {}
            if not os.path.exists(pred_path) or RECALCULATE_PREDICTION:
                y_pred, meta = func[0](
                    y_proba,
                    k,
                    priors=priors,
                    inv_ps=inv_ps,
                    seed=seed,
                    **func[1],
                )

                #print(y_proba[-1], y_proba_copy[-1])

                results.update(meta)
                print(f"  Iters: {meta['iters']}")
                print(f"  Time: {meta['time']:>5.2f} s")
                #save_npz_wrapper(pred_path, y_pred)
                save_json(results_path, results)
            else:
                #y_pred = load_npz_wrapper(pred_path)
                results = load_json(results_path)

            print("  Metrics (%):")
            results.update(calculate_and_report_metrics(y_true, y_pred, k, METRICS))
            save_json(results_path, results)

        print("  Done")


if __name__ == "__main__":
    main()
