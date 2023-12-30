import sys

import click
import numpy as np
from data import *

from columns.block_coordinate import *
from columns.metrics import *
from columns.online_block_coordinate import *
from columns.weighted_prediction import *
from columns.find_classifier_frank_wolfe import *


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

def frank_wolfe_wrapper(
    y_proba,
    utility_func,
    k: int = 5,
    seed: int = 0,
    reg=0,
    pred_repeat=10,
    average=False,
    use_last=False,
    y_true_valid=None, y_proba_valid=None,
    **kwargs,
):
    classifiers, classifier_weights, meta = find_classifier_frank_wolfe(
        y_true_valid, y_proba_valid, utility_func, max_iters=20, k=k, reg=reg, **kwargs
    )
    print(f"  classifiers weights: {classifier_weights}")
    y_preds = []
    if use_last:
        print("  using last classifier")
        y_pred = predict_top_k_for_classfiers(
            y_proba, classifiers[-1:], np.array([1]), k=k, seed=seed
        )
        y_preds.append(y_pred)
    elif not average:
        print("  predicting with randomized classfier")
        for i in range(pred_repeat):
            y_pred = predict_top_k_for_classfiers(
                y_proba, classifiers, classifier_weights, k=k, seed=seed + i
            )
            y_preds.append(y_pred)
    else:
        print("  averaging classifiers weights")
        avg_classifier_weights = np.zeros((classifiers.shape[1], classifiers.shape[2]))
        for i in range(classifier_weights.shape[0]):
            avg_classifier_weights += classifier_weights[i] * classifiers[i]
        avg_classifier_weights /= classifier_weights.shape[0]
        y_pred = predict_top_k(y_proba, avg_classifier_weights, k)
        y_preds.append(y_pred)
    
    return y_preds[0], meta


def fw_macro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true_valid=None, y_preds_valid=None, **kwargs
):
    return frank_wolfe_wrapper(
        y_proba, macro_f1_C, y_true_valid=y_true_valid, y_preds_valid=y_preds_valid, k=k, seed=seed, **kwargs
    )


METHODS = {
    "online-block-coord-macro-f1": (online_bc_macro_f1, {}),
    "online-greedy-block-coord-macro-f1": (online_bc_macro_f1, {"greedy": True, "num_valid_sets": 1, "valid_set_size": 1}),
    "frank-wolfe-macro-f1": (fw_macro_f1, {}),
    "frank-wolfe-average-macro-f1": (fw_macro_f1, {"average": True}),
    "frank-wolfe-last-macro-f1": (fw_macro_f1, {"use_last": True}),

    # Instance-wise measures / baselines
    # "random": (predict_random_at_k,{}),
    "optimal-instance-prec": (predict_for_optimal_instance_precision, {}),
    "block-coord-instance-prec": (bc_instance_precision_at_k, {}),
    "optimal-instance-ps-prec": (inv_propensity_weighted_instance, {}),
    # "power-law-with-beta=0.875": (power_law_weighted_instance, {"beta": 0.875}),
    "power-law-with-beta=0.75": (power_law_weighted_instance, {"beta": 0.75}),
    "power-law-with-beta=0.5": (power_law_weighted_instance, {"beta": 0.5}),
    "power-law-with-beta=0.25": (power_law_weighted_instance, {"beta": 0.25}),
    # "power-law-with-beta=0.125": (power_law_weighted_instance, {"beta": 0.125}),
    "log": (predict_log_weighted_per_instance, {}),
    "optimal-macro-recall": (predict_for_optimal_macro_recall, {}),

    # Block coordinate with default parameters - commented out because it better to use variatns with specific tolerance to stopping condition
    # "block-coord-macro-prec": (bc_macro_precision, {}),
    # "block-coord-macro-recall": (bc_macro_recall, {}),
    # "block-coord-macro-f1": (bc_macro_f1, {}),
    # "block-coord-cov": (bc_coverage, {}),
    
    # Greedy / 1 iter variants
    # "greedy-macro-prec": (bc_macro_precision, {"init_y_pred": "greedy", "max_iter": 1}),
    # "greedy-macro-recall": (bc_macro_precision, {"init_y_pred": "greedy", "max_iter": 1}),
    "greedy-macro-f1": (bc_macro_f1, {"init_y_pred": "greedy", "max_iter": 1}),
    # "greedy-cov": (bc_coverage, {"init_y_pred": "greedy", "max_iter": 1}),
    # "block-coord-macro-prec-iter=1": (bc_macro_precision, {"max_iter": 1}),
    # "block-coord-macro-recall-iter=1": (bc_macro_precision, {"max_iter": 1}),
    "block-coord-macro-f1-iter=1": (bc_macro_f1, {"max_iter": 1}),
    # "block-coord-cov-iter=1": (bc_coverage, {"max_iter": 1}),
    # "greedy-start-block-coord-macro-prec": (bc_macro_precision, {"init_y_pred": "greedy"}),
    # "greedy-start-block-coord-macro-recall": (bc_macro_f1, {"init_y_pred": "greedy"}),
    # "greedy-start-block-coord-macro-f1": (bc_macro_f1, {"init_y_pred": "greedy"}),
    # "greedy-start-block-coord-cov": (bc_coverage, {"init_y_pred": "greedy"}),

    # Tolerance on stopping condiction experiments
    #"block-coord-macro-prec-tol=1e-3": (bc_macro_precision, {"tolerance": 1e-3}),
    #"block-coord-macro-prec-tol=1e-4": (bc_macro_precision, {"tolerance": 1e-4}),
    "block-coord-macro-prec-tol=1e-5": (bc_macro_precision, {"tolerance": 1e-5}),
    "block-coord-macro-prec-tol=1e-6": (bc_macro_precision, {"tolerance": 1e-6}),
    "block-coord-macro-prec-tol=1e-7": (bc_macro_precision, {"tolerance": 1e-7}),

    # For recall all should be the same
    #"block-coord-macro-recall-tol=1e-3": (bc_macro_recall, {"tolerance": 1e-3}),
    #"block-coord-macro-recall-tol=1e-4": (bc_macro_recall, {"tolerance": 1e-4}),
    "block-coord-macro-recall-tol=1e-5": (bc_macro_recall, {"tolerance": 1e-5}),
    "block-coord-macro-recall-tol=1e-6": (bc_macro_recall, {"tolerance": 1e-6}),
    "block-coord-macro-recall-tol=1e-7": (bc_macro_recall, {"tolerance": 1e-7}),

    #"block-coord-macro-f1-tol=1e-3": (bc_macro_f1, {"tolerance": 1e-3}),
    #"block-coord-macro-f1-tol=1e-4": (bc_macro_f1, {"tolerance": 1e-4}),
    "block-coord-macro-f1-tol=1e-5": (bc_macro_f1, {"tolerance": 1e-5}),
    "block-coord-macro-f1-tol=1e-6": (bc_macro_f1, {"tolerance": 1e-6}),
    "block-coord-macro-f1-tol=1e-7": (bc_macro_f1, {"tolerance": 1e-7}),
    
    #"block-coord-cov-tol=1e-3": (bc_coverage, {"tolerance": 1e-3}),
    #"block-coord-cov-tol=1e-4": (bc_coverage, {"tolerance": 1e-4}),
    "block-coord-cov-tol=1e-5": (bc_coverage, {"tolerance": 1e-5}),
    "block-coord-cov-tol=1e-6": (bc_coverage, {"tolerance": 1e-6}),
    "block-coord-cov-tol=1e-7": (bc_coverage, {"tolerance": 1e-7}),
}

# Add variants with different alpha for mixed utilities
# alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
# alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999]
# for alpha in alphas:
#     METHODS[f"block-coord-mixed-prec-f1-alpha={alpha}-tol=1e-6"] = (bc_mixed_instance_prec_macro_f1, {"alpha": alpha, "tolerance": 1e-6})
#     METHODS[f"block-coord-mixed-prec-prec-alpha={alpha}-tol=1e-6"] = (bc_mixed_instance_prec_macro_prec, {"alpha": alpha, "tolerance": 1e-6})
#     METHODS[f"block-coord-mixed-prec-cov-alpha={alpha}-tol=1e-6"] = (bc_coverage, {"alpha": alpha, "tolerance": 1e-6})
#     METHODS[f"block-coord-mixed-prec-f1-alpha={alpha}-tol=1e-7"] = (bc_mixed_instance_prec_macro_f1, {"alpha": alpha, "tolerance": 1e-7})
#     METHODS[f"block-coord-mixed-prec-prec-alpha={alpha}-tol=1e-7"] = (bc_mixed_instance_prec_macro_prec, {"alpha": alpha, "tolerance": 1e-7})
#     METHODS[f"block-coord-mixed-prec-cov-alpha={alpha}-tol=1e-7"] = (bc_coverage, {"alpha": alpha, "tolerance": 1e-7})


def calculate_and_report_metrics(y_true, y_pred, k, metrics):
    results = {}
    print(y_true.shape, y_pred.shape)
    for metric, func in metrics.items():
        value = func(y_true, y_pred)
        results[f"{metric}@{k}"] = value
        print(f"    {metric}@{k}: {100 * value:>5.2f}")

    return results


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("-k", type=int, required=True)
@click.option("-s", "--seed", type=int, required=True)
@click.option("-m", "--method", type=str, required=False, default=None)
@click.option("-p", "--probabilities_path", type=str, required=False, default=None)
@click.option("-l", "--labels_path", type=str, required=False, default=None)
@click.option("-r", "--results_dir", type=str, required=False, default="results_bca")
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
    if "yeast_plt" in experiment:
        # yeast - PLT
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/yeast_top_200_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/yeast/yeast_test.txt",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/yeast/yeast_train.txt",
            "load_func": load_txt_labels,
        }

    elif "youtube_deepwalk_plt" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/youtube_deepwalk_top_200_{plt_loss}",
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

    elif "eurlex_lexglue_plt" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/eurlex_lexglue_top_200_{plt_loss}",
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

    elif "mediamill_plt" in experiment:
        # mediamill - PLT
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/mediamill_top_200_{plt_loss}",
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

    elif "flicker_deepwalk_plt" in experiment:
        xmlc_data_load_config["header"] = False
        y_proba_path = {
            "path": f"predictions/flicker_deepwalk_top_200_{plt_loss}",
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

    elif "rcv1x_plt" in experiment:
        # RCV1X - PLT + XMLC repo data
        y_proba_path = {
            "path": f"predictions/rcv1x_top_200_{plt_loss}",
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

    elif "eurlex_plt" in experiment:
        # Eurlex - PLT + XMLC repo data
        y_proba_path = {
            "path": f"predictions/eurlex_top_200_{plt_loss}",
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

    elif "eurlex2_plt" in experiment:
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
            "path": f"predictions/amazonCat_top_200_{plt_loss}",
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

    elif "wiki10_plt" in experiment:
        # Wiki10 - PLT + XMLC repo data
        y_proba_path = {
            "path": f"predictions/wiki10_top_200_{plt_loss}",
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

    elif "wiki102_plt" in experiment:
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

    elif "amazon_plt" in experiment:
        # Amazon - PLT + XMLC repo data
        y_proba_path = {
            "path": f"predictions/amazon_top_200_{plt_loss}",
            "load_func": load_txt_sparse_pred,
        }
        y_true_path = {
            "path": "datasets/amazon/amazon_test.txt",
            "load_func": load_txt_labels,
        }
        train_y_true_path = {
            "path": "datasets/amazon/amazon_train.txt",
            "load_func": load_txt_labels,
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

    # Calculate marginals and propensities
    with Timer():
        print("Calculating marginals and propensities")
        marginals = labels_priors(train_y_true)
        # marginals = labels_priors(y_true)
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
    # y_true = y_true.toarray()
    # y_proba = y_proba.toarray()
    # train_y_true = train_y_true.toarray()

    output_path_prefix = f"results_bca4/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for method, func in methods.items():
        print(f"{experiment} - {method} @ {k}: ")

        output_path = f"{output_path_prefix}{method}_k={k}_s={seed}"
        results_path = f"{output_path}_results.json"
        pred_path = f"{output_path}_pred.npz"

        func[1]["return_meta"] = True  # Include meta data in results
        func[1]["y_true_train"] = train_y_true
        func[1]["y_proba_train"] = train_y_proba
        func[1]["y_true_valid"] = train_y_true
        func[1]["y_proba_valid"] = train_y_proba
        if not os.path.exists(results_path) or RECALCULATE_RESUTLS:
            results = {}
            if not os.path.exists(pred_path) or RECALCULATE_PREDICTION:
                y_pred, meta = func[0](
                    y_proba,
                    k,
                    marginals=marginals,
                    inv_ps=inv_ps,
                    seed=seed,
                    **func[1],
                )
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
