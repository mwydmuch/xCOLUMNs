import inspect
import sys

import click
import numpy as np

from utils import *
from xcolumns.block_coordinate import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics_original import *
from xcolumns.weighted_prediction import *


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
    # Instance-wise measures / baselines
    "optimal-instance-prec": (predict_optimizing_instance_precision, {}),
    # "block-coord-instance-prec": (bc_instance_precision_at_k, {}), # This is the same as optimal-instance-prec but using block coordinate, for sanity-check purposes only
    "optimal-instance-ps-prec": (
        predict_optimizing_instance_propensity_scored_precision,
        {},
    ),
    "power-law-with-beta=0.75": (
        predict_power_law_weighted_per_instance,
        {"beta": 0.75},
    ),
    "power-law-with-beta=0.5": (predict_power_law_weighted_per_instance, {"beta": 0.5}),
    "power-law-with-beta=0.25": (
        predict_power_law_weighted_per_instance,
        {"beta": 0.25},
    ),
    "log": (predict_log_weighted_per_instance, {}),
    "optimal-macro-recall": (predict_optimizing_macro_recall, {}),
    #
    # Block coordinate with default parameters - commented out because it better to use variatns with specific tolerance to stopping condition
    # "block-coord-macro-prec": (predict_optimizing_macro_precision_using_bc, {}),
    # "block-coord-macro-recall": (predict_optimizing_macro_recall_using_bc, {}),
    # "block-coord-macro-f1": (predict_optimizing_macro_f1_score_using_bc, {}),
    # "block-coord-cov": (predict_optimizing_coverage_using_bc, {}),
    #
    # Tolerance on stopping condiction experiments
    "block-coord-macro-prec-tol=1e-3": (
        predict_optimizing_macro_precision_using_bc,
        {"tolerance": 1e-3},
    ),
    "block-coord-macro-prec-tol=1e-4": (
        predict_optimizing_macro_precision_using_bc,
        {"tolerance": 1e-4},
    ),
    "block-coord-macro-prec-tol=1e-5": (
        predict_optimizing_macro_precision_using_bc,
        {"tolerance": 1e-5},
    ),
    "block-coord-macro-prec-tol=1e-6": (
        predict_optimizing_macro_precision_using_bc,
        {"tolerance": 1e-6},
    ),
    "block-coord-macro-prec-tol=1e-7": (
        predict_optimizing_macro_precision_using_bc,
        {"tolerance": 1e-7},
    ),
    #
    # For recall all should be the same
    "block-coord-macro-recall-tol=1e-3": (
        predict_optimizing_macro_recall_using_bc,
        {"tolerance": 1e-3},
    ),
    "block-coord-macro-recall-tol=1e-4": (
        predict_optimizing_macro_recall_using_bc,
        {"tolerance": 1e-4},
    ),
    "block-coord-macro-recall-tol=1e-5": (
        predict_optimizing_macro_recall_using_bc,
        {"tolerance": 1e-5},
    ),
    "block-coord-macro-recall-tol=1e-6": (
        predict_optimizing_macro_recall_using_bc,
        {"tolerance": 1e-6},
    ),
    "block-coord-macro-recall-tol=1e-7": (
        predict_optimizing_macro_recall_using_bc,
        {"tolerance": 1e-7},
    ),
    #
    "block-coord-macro-f1-tol=1e-3": (
        predict_optimizing_macro_f1_score_using_bc,
        {"tolerance": 1e-3},
    ),
    "block-coord-macro-f1-tol=1e-4": (
        predict_optimizing_macro_f1_score_using_bc,
        {"tolerance": 1e-4},
    ),
    "block-coord-macro-f1-tol=1e-5": (
        predict_optimizing_macro_f1_score_using_bc,
        {"tolerance": 1e-5},
    ),
    "block-coord-macro-f1-tol=1e-6": (
        predict_optimizing_macro_f1_score_using_bc,
        {"tolerance": 1e-6},
    ),
    "block-coord-macro-f1-tol=1e-7": (
        predict_optimizing_macro_f1_score_using_bc,
        {"tolerance": 1e-7},
    ),
    #
    "block-coord-cov-tol=1e-3": (
        predict_optimizing_coverage_using_bc,
        {"tolerance": 1e-3},
    ),
    "block-coord-cov-tol=1e-4": (
        predict_optimizing_coverage_using_bc,
        {"tolerance": 1e-4},
    ),
    "block-coord-cov-tol=1e-5": (
        predict_optimizing_coverage_using_bc,
        {"tolerance": 1e-5},
    ),
    "block-coord-cov-tol=1e-6": (
        predict_optimizing_coverage_using_bc,
        {"tolerance": 1e-6},
    ),
    "block-coord-cov-tol=1e-7": (
        predict_optimizing_coverage_using_bc,
        {"tolerance": 1e-7},
    ),
    # Greedy / 1 iter variants
    "greedy-macro-prec": (
        predict_optimizing_macro_precision_using_bc,
        {"init_y_pred": "greedy", "max_iters": 1},
    ),
    "greedy-macro-recall": (
        predict_optimizing_macro_precision_using_bc,
        {"init_y_pred": "greedy", "max_iters": 1},
    ),
    "greedy-macro-f1": (
        predict_optimizing_macro_f1_score_using_bc,
        {"init_y_pred": "greedy", "max_iters": 1},
    ),
    "greedy-cov": (
        predict_optimizing_coverage_using_bc,
        {"init_y_pred": "greedy", "max_iters": 1},
    ),
    #
    "block-coord-macro-prec-iter=1": (
        predict_optimizing_macro_precision_using_bc,
        {"max_iters": 1},
    ),
    "block-coord-macro-recall-iter=1": (
        predict_optimizing_macro_precision_using_bc,
        {"max_iters": 1},
    ),
    "block-coord-macro-f1-iter=1": (
        predict_optimizing_macro_f1_score_using_bc,
        {"max_iters": 1},
    ),
    "block-coord-cov-iter=1": (predict_optimizing_coverage_using_bc, {"max_iters": 1}),
    #
    # Similar results to the above
    # "greedy-start-block-coord-macro-prec": (predict_optimizing_macro_precision_using_bc, {"init_y_pred": "greedy"},),
    # "greedy-start-block-coord-macro-recall": (predict_optimizing_macro_f1_score_using_bc, {"init_y_pred": "greedy"}),
    # "greedy-start-block-coord-macro-f1": (predict_optimizing_macro_f1_score_using_bc, {"init_y_pred": "greedy"}),
    # "greedy-start-block-coord-cov": (predict_optimizing_coverage_using_bc, {"init_y_pred": "greedy"}),
}

# Add variants with different alpha for mixed utilities
alphas = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.995,
    0.999,
]
for alpha in alphas:
    pass
    METHODS[f"block-coord-mixed-prec-f1-alpha={alpha}-tol=1e-6"] = (
        predict_optimizing_mixed_instance_precision_and_macro_f1_score_using_bc,
        {"alpha": alpha, "tolerance": 1e-6},
    )
    METHODS[f"block-coord-mixed-prec-prec-alpha={alpha}-tol=1e-6"] = (
        predict_optimizing_mixed_instance_precision_and_macro_precision_using_bc,
        {"alpha": alpha, "tolerance": 1e-6},
    )
    METHODS[f"block-coord-mixed-prec-cov-alpha={alpha}-tol=1e-6"] = (
        predict_optimizing_coverage_using_bc,
        {"alpha": alpha, "tolerance": 1e-6},
    )
    METHODS[f"power-law-with-beta={alpha}"] = (
        predict_power_law_weighted_per_instance,
        {"beta": alpha},
    )


def calculate_and_report_metrics(y_true, y_pred, k, metrics):
    results = {}
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
@click.option("-r", "--results_dir", type=str, required=False, default="results_bc/")
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
    if method is None:
        methods = METHODS
    elif method in METHODS:
        methods = {method: METHODS[method]}
    else:
        raise ValueError(f"Unknown method: {method}")

    true_as_pred = "true_as_pred" in experiment
    lightxml_data = "lightxml" in experiment

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
    train_y_true = None

    y_proba_path = {
        "path": probabilities_path,
        "load_func": load_txt_sparse_pred,
    }
    y_true_path = {
        "path": labels_path,
        "load_func": load_txt_labels,
    }

    # Predefined experiments
    if "eurlex_lightxml" in experiment:
        # Eurlex - LightXML
        y_true_path = {
            "path": "datasets/EUR-Lex/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/lightxml/eurlex/eurlex4k_full_plain-scores.npy",
            "load_func": load_npy_full_pred,
            "keep_top_k": 100,
            "apply_sigmoid": True,
        }
        train_y_true_path = {
            "path": "datasets/EUR-Lex/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "amazoncat_100_lightxml" in experiment:
        # Wiki - LightXML
        y_true_path = {
            "path": "datasets/AmazonCat-13K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/lightxml/amazonCat_100/amazonCat.notnpz",
            "load_func": load_npz_wrapper,
            "apply_sigmoid": True,
        }
        train_y_true_path = {
            "path": "datasets/AmazonCat-13K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "wiki10_100_lightxml" in experiment:
        # Wiki - LightXML
        y_true_path = {
            "path": "datasets/Wiki10-31K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/lightxml/wiki10_100/wiki10.notnpz",
            "load_func": load_npz_wrapper,
            "apply_sigmoid": True,
        }
        train_y_true_path = {
            "path": "datasets/Wiki10-31K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "amazon_100_lightxml" in experiment:
        # Amazon - LightXML
        y_true_path = {
            "path": "datasets/Amazon-670K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/lightxml/amazon_100/amazon670k_light_t0",
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
            "path": "predictions/lightxml/amazon_1000/amazon670k_light_original_t0",
            "load_func": load_npy_sparse_pred,
        }
        train_y_true_path = {
            "path": "datasets/Amazon-670K/train_labels.txt",
            "load_func": load_txt_labels,
        }

    elif "wiki500_100_lightxml" in experiment:
        # WikiLarge - LightXML
        y_true_path = {
            "path": "datasets/Wiki-500K/test_labels.txt",
            "load_func": load_txt_labels,
        }
        y_proba_path = {
            "path": "predictions/lightxml/wiki500_100/wiki500k_light_t0",
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
            "path": "predictions/lightxml/wiki500_1000/wiki500k_light_original_t0",
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

    # Convert to array to check if it gives the same results
    if use_dense:
        y_true = y_true.toarray()
        y_proba = y_proba.toarray()
        train_y_true = train_y_true.toarray()

    output_path_prefix = f"{results_dir}/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for method, func in methods.items():
        print(f"{experiment} - {method} @ {k} (seed {seed}): ")

        output_path = f"{output_path_prefix}{method}_k={k}_s={seed}"
        results_path = f"{output_path}_results.json"
        pred_path = f"{output_path}_pred.npz"

        if (
            not os.path.exists(results_path)
            or recalculate_results
            or recalculate_predictions
        ):
            results = {}
            if not os.path.exists(pred_path) or recalculate_predictions:
                y_pred, meta = call_function_with_supported_kwargs(
                    func[0],
                    y_proba,
                    k,
                    priors=priors,
                    inverse_propensities=inv_ps,
                    seed=seed,
                    return_meta=True,
                    verbose=True,
                    **func[1],
                )
                results.update(meta)
                print(f"  Iters: {meta['iters']}")
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
