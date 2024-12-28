import inspect
import sys

import click
import numpy as np
import scipy.sparse as sp
from metrics_original import *
from sklearn.model_selection import train_test_split
from thesis_frank_wolfe_wrappers import *

from utils import *
from xcolumns.block_coordinate import *
from xcolumns.confusion_matrix import *
from xcolumns.frank_wolfe import *
from xcolumns.weighted_prediction import *


METRICS_ON_Y = {
    "mC": macro_abandonment_v0,
    "iC": instance_abandonment_v0,
    "mP": macro_precision_v0,
    "iP": instance_precision_v0,
    "mR": macro_recall_v0,
    "iR": instance_recall_v0,
    "mF": macro_f1_v0,
    "iF": instance_f1_v0,
    "iBA": instance_balanced_accuracy_v0,
    "mBA": macro_balanced_accuracy_v0,
    "instance-recall": instance_recall,
    "instance-precision": instance_precision,
    "instance-f1": instance_f1_score,
    "instance-balanced-accuracy": instance_balanced_accuracy,
    "abandonment": abandonment,
}
METRICS_ON_C_MAT = {
    "precision-at-k": precision_at_k_on_conf_matrix,
    "ps-precision": weighted_precision_at_k_on_conf_matrix,
    "macro-precision": macro_precision_on_conf_matrix,
    "macro-recall": macro_recall_on_conf_matrix,
    "macro-f1": macro_f1_score_on_conf_matrix,
    "coverage": coverage_on_conf_matrix,
    "macro-jaccard-score": macro_jaccard_score_on_conf_matrix,
    "macro-balanced-accuracy": macro_balanced_accuracy_on_conf_matrix,
    "macro-gmean": macro_gmean_on_conf_matrix,
    "macro-hmean": macro_hmean_on_conf_matrix,
}


def calculate_and_report_metrics(y_true, y_pred, k=5, inverse_propensities=None):
    results = {}
    for metric, func in METRICS_ON_Y.items():
        value = func(y_true, y_pred)
        results[f"{metric}@{k}"] = value
        print(f"    {metric}@{k}: {100 * value:>5.3f}")

    conf_mat = calculate_confusion_matrix(y_true, y_pred, normalize=True)
    # print(*conf_mat)
    for metric, func in METRICS_ON_C_MAT.items():
        value = call_function_with_supported_kwargs(
            func, *conf_mat, k=k, w=inverse_propensities
        )
        results[f"{metric}@{k}"] = value
        print(f"    {metric}@{k}: {100 * value:>5.3f}")

    # conf_mat = calculate_confusion_matrix(y_true, y_pred, normalize=False)
    # for metric, func in METRICS_ON_C_MAT.items():
    #     value = call_function_with_supported_kwargs(
    #         func,
    #         *conf_mat,
    #         k=k
    #     )
    #     results[f"non-norm-{metric}@{k}"] = value
    #     print(f"    non-norm-{metric}@{k}: {100 * value:>5.3f}")

    return results


TOL = 1e-7
METHODS = {
    # Instance-wise measures / baselines
    "optimal-instance-precision": (predict_optimizing_instance_precision, {}),
    # "block-coord-instance-precision": (bc_instance_precision_at_k, {}), # This is the same as optimal-instance-precision but using block coordinate, for sanity-check purposes only
    "optimal-instance-ps-precision": (
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
    "optimal-macro-balanced-accuracy": (predict_optimizing_macro_balanced_accuracy, {}),
    #
    # Block coordinate with default parameters - commented out because it better to use variatns with specific tolerance to stopping condition
    # "block-coord-macro-precision": (predict_optimizing_macro_precision_using_bc, {}),
    # "block-coord-macro-recall": (predict_optimizing_macro_recall_using_bc, {}),
    # "block-coord-macro-f1": (predict_optimizing_macro_f1_score_using_bc, {}),
    # "block-coord-coverage": (predict_optimizing_coverage_using_bc, {}),
    #
    # Tolerance on stopping condiction experiments
    f"block-coord-macro-precision-tol={TOL}": (
        predict_optimizing_macro_precision_using_bc,
        {"tolerance": TOL},
    ),
    f"block-coord-macro-recall-tol={TOL}": (
        predict_optimizing_macro_recall_using_bc,
        {"tolerance": TOL},
    ),
    f"block-coord-macro-f1-tol={TOL}": (
        predict_optimizing_macro_f1_score_using_bc,
        {"tolerance": TOL},
    ),
    f"block-coord-coverage-tol={TOL}": (
        predict_optimizing_coverage_using_bc,
        {"tolerance": TOL},
    ),
    f"block-coord-macro-jaccard-score-tol={TOL}": (
        predict_optimizing_macro_jaccard_score_using_bc,
        {"tolerance": TOL},
    ),
    f"block-coord-macro-balanced-accuracy-tol={TOL}": (
        predict_optimizing_macro_balanced_accuracy_using_bc,
        {"tolerance": TOL},
    ),
    # f"block-coord-macro-gmean-tol={TOL}": (
    #     predict_optimizing_macro_gmean_using_bc,
    #     {"tolerance": TOL},
    # ),
    # f"block-coord-macro-hmean-tol={TOL}": (
    #     predict_optimizing_macro_hmean_using_bc,
    #     {"tolerance": TOL},
    # ),
    # Greedy / 1 iter variants
    "greedy-macro-precision": (
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
    "greedy-coverage": (
        predict_optimizing_coverage_using_bc,
        {"init_y_pred": "greedy", "max_iters": 1},
    ),
    "greedy-macro-jaccard-score": (
        predict_optimizing_macro_jaccard_score_using_bc,
        {"init_y_pred": "greedy", "max_iters": 1},
    ),
    "greedy-macro-balanced-accuracy": (
        predict_optimizing_macro_balanced_accuracy_using_bc,
        {"init_y_pred": "greedy", "max_iters": 1},
    ),
}

for args in [{}, {"max_iters": 1}]:
    METHODS.update(
        {
            # Frank wolfe,
            "frank-wolfe-macro-precision": (
                find_and_predict_for_macro_precision_using_fw,
                {},
            ),
            "frank-wolfe-macro-recall": (
                find_and_predict_for_macro_recall_using_fw,
                {},
            ),
            "frank-wolfe-macro-f1": (find_and_predict_for_macro_f1_score_using_fw, {}),
            "frank-wolfe-macro-jaccard-score": (
                find_and_predict_for_macro_jaccard_score_using_fw,
                {},
            ),
            "frank-wolfe-macro-balanced-accuracy": (
                find_and_predict_for_macro_balanced_accuracy_using_fw,
                {},
            ),
            "frank-wolfe-macro-gmean": (find_and_predict_for_macro_gmean_using_fw, {}),
            "frank-wolfe-macro-hmean": (find_and_predict_for_macro_hmean_using_fw, {}),
        }
    )


# Add variants with different alpha for mixed utilities
alphas = [
    # 0.01,
    # 0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.7,
    0.8,
    0.9,
    # 0.95,
    # 0.99,
    # 0.995,
    # 0.999,
]
alphas = [0, 0.01, 0.5, 0.99, 1]
alphas = []
for alpha in alphas:
    METHODS[f"block-coord-mixed-precision-macro-f1-alpha={alpha}-tol={TOL}"] = (
        predict_optimizing_mixed_instance_precision_and_macro_f1_score_using_bc,
        {"alpha": alpha, "tolerance": TOL},
    )
    METHODS[f"block-coord-mixed-precision-macro-precision-alpha={alpha}-tol={TOL}"] = (
        predict_optimizing_mixed_instance_precision_and_macro_precision_using_bc,
        {"alpha": alpha, "tolerance": TOL},
    )
    METHODS[f"block-coord-mixed-precision-macro-coverage-alpha={alpha}-tol={TOL}"] = (
        predict_optimizing_coverage_using_bc,
        {"alpha": alpha, "tolerance": TOL},
    )
    METHODS[
        f"block-coord-mixed-precision-macro-jaccard-score-alpha={alpha}-tol={TOL}"
    ] = (
        predict_optimizing_mixed_instance_precision_and_macro_jaccard_score_using_bc,
        {"alpha": alpha, "tolerance": TOL},
    )
    METHODS[
        f"block-coord-mixed-precision-macro-balanced-accuracy-alpha={alpha}-tol={TOL}"
    ] = (
        predict_optimizing_mixed_instance_precision_and_macro_balanced_accuracy_using_bc,
        {"alpha": alpha, "tolerance": TOL},
    )
    # METHODS[f"block-coord-mixed-precision-macro-gmean-alpha={alpha}-tol={TOL}"] = (
    #     predict_optimizing_mixed_instance_precision_and_macro_gmean_accuracy_using_bc,
    #     {"alpha": alpha, "tolerance": TOL},
    # )
    # METHODS[f"block-coord-mixed-precision-macro-hmean-alpha={alpha}-tol={TOL}"] = (
    #     predict_optimizing_mixed_instance_precision_and_macro_hmean_accuracy_using_bc,
    #     {"alpha": alpha, "tolerance": TOL},
    # )
    METHODS[f"power-law-with-beta={alpha}"] = (
        predict_power_law_weighted_per_instance,
        {"beta": alpha},
    )

    METHODS[f"frank-wolfe-mixed-precision-macro-precision-alpha={alpha}"] = (
        find_and_predict_for_mixed_instance_precision_and_macro_precision_using_fw,
        {"alpha": alpha},
    )

    METHODS[f"frank-wolfe-mixed-macro-recall-macro-precision-alpha={alpha}"] = (
        find_and_predict_for_mixed_macro_recall_and_macro_precision_using_fw,
        {"alpha": alpha},
    )


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("-k", type=int, required=True)
@click.option("-s", "--seed", type=int, required=True)
@click.option("-m", "--method", type=str, required=False, default=None)
@click.option("-p", "--probabilities_path", type=str, required=False, default=None)
@click.option("-l", "--labels_path", type=str, required=False, default=None)
@click.option(
    "-r", "--results_dir", type=str, required=False, default="results_thesis_sun"
)
@click.option(
    "--recalculate_predictions", is_flag=True, type=bool, required=False, default=False
)
@click.option(
    "--recalculate_results", is_flag=True, type=bool, required=False, default=False
)
@click.option("--test_multiply", type=int, required=False, default=1)
@click.option("--use_proba_as_true", is_flag=True, required=False, default=False)
@click.option("--use_train_as_test", is_flag=True, required=False, default=False)
@click.option("--use_dense", is_flag=True, type=bool, required=False, default=False)
@click.option("-v", "--val_split", type=float, required=False, default=0.0)
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
    test_multiply,
    use_proba_as_true,
    use_train_as_test,
    use_dense,
    val_split,
):
    if method is None:
        methods = METHODS
    elif method in METHODS:
        methods = {method: METHODS[method]}
    else:
        methods = {k: v for k, v in METHODS.items() if method in k}
    if len(methods) == 0:
        raise ValueError(f"Unknown method: {method}")

    use_true_as_pred = "use_true_as_pred" in experiment
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
    train_y_proba_path = None
    train_y_proba = None

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
            "path": "predictions/lightxml/amazoncat_100/amazonCat.notnpz",
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

    nxc_datasets = [
        "rcv1x",
        "eurlex",
        "amazonCat",
        "wiki10",
        "amazon",
        "amazon-3M",
        "wikiLSHTC",
        "WikipediaLarge-500K",
        "amazonCat-14K",
        "deliciousLarge",
    ]
    for d in nxc_datasets:
        if f"{d}_100_plt" in experiment:
            y_proba_path = {
                "path": f"predictions/nxc/{d}/test_pred",
                "load_func": load_txt_sparse_pred,
            }
            y_true_path = {
                "path": f"datasets/{d}/{d}_test.txt",
                "load_func": load_txt_labels,
            }
            train_y_true_path = {
                "path": f"datasets/{d}/{d}_train.txt",
                "load_func": load_txt_labels,
            }
            if os.path.exists(f"predictions/nxc/{d}/train_pred"):
                train_y_proba_path = {
                    "path": f"predictions/nxc/{d}/train_pred",
                    "load_func": load_txt_sparse_pred,
                }
            break

    # else:
    #     raise RuntimeError(f"No matching experiment: {experiment}")

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

    if train_y_true.shape != y_true.shape:
        align_dim1(y_true, train_y_true)

    # For some sparse format this resize might be necessary
    if y_true.shape != y_proba.shape:
        if y_true.shape[0] != y_proba.shape[0]:
            raise RuntimeError(
                f"Number of instances in true and prediction do not match {y_true.shape[0]} != {y_proba.shape[0]}"
            )
        align_dim1(y_true, y_proba)

    val_y_proba = None
    val_y_true = None
    if val_split > 0:
        y_true, val_y_true, y_proba, val_y_proba = train_test_split(
            y_true, y_proba, test_size=val_split, random_state=seed
        )
    if train_y_true is not None and train_y_proba is not None:
        val_y_proba = train_y_proba
        val_y_true = train_y_true

    # Calculate priors and propensities
    with Timer():
        print("Calculating priors and propensities")
        if val_y_true is not None:
            priors = labels_priors(val_y_true)
        else:
            priors = labels_priors(train_y_true)
        inv_ps = jpv_inverse_propensity(train_y_true)

    if "apply_sigmoid" in y_proba_path and y_proba_path["apply_sigmoid"]:
        # LightXML predictions aren't probabilities
        y_proba.data = 1.0 / (1.0 + np.exp(-y_proba.data))

    # Use true labels as predictions with 1.0 score (probability)
    if use_true_as_pred:
        y_proba = y_true

    if use_proba_as_true:
        y_true = y_proba
        if val_y_proba is not None and val_y_true is not None:
            val_y_true = val_y_proba
        if train_y_proba is not None and train_y_true is not None:
            train_y_true = train_y_proba

    if use_train_as_test:
        y_true = train_y_true
        y_proba = train_y_proba
        if val_y_proba is not None and val_y_true is not None:
            val_y_true = train_y_true
            val_y_proba = train_y_proba

    def repeat_csr_matrix(matrix, n):
        return sp.vstack([matrix] * n)

    if test_multiply > 1:
        y_true = sp.vstack([y_true] * test_multiply)
        y_proba = sp.vstack([y_proba] * test_multiply)

    print(f"y_true: type={type(y_true)}, shape={y_true.shape}")
    print(f"y_proba: type={type(y_proba)}, shape={y_proba.shape}")
    if train_y_true is not None:
        print(f"train_y_true: type={type(train_y_true)}, shape={train_y_true.shape}")

    # Convert to array to check if it gives the same results
    if use_dense:
        y_true = y_true.toarray()
        y_proba = y_proba.toarray()
        train_y_true = train_y_true.toarray()

    # print(y_pred.shape, inv_ps.shape, priors.shape, y_true.shape, y_true_test.shape, y_proba.shape, y_proba_test.shape)

    output_path_prefix = f"{results_dir}/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for method, func in methods.items():
        print(f"{experiment} - {method} @ {k} (seed {seed}): ")

        output_path = f"{output_path_prefix}{method}_k={k}_v={val_split}_s={seed}"
        results_path = f"{output_path}_results.json"
        pred_path = f"{output_path}_pred.npz"

        if (
            not os.path.exists(results_path)
            or recalculate_results
            or recalculate_predictions
        ):
            results = {}
            if not os.path.exists(pred_path) or recalculate_predictions:
                try:
                    y_pred, meta = call_function_with_supported_kwargs(
                        func[0],
                        y_proba,
                        k,
                        val_y_true=val_y_true,
                        val_y_proba=val_y_proba,
                        priors=priors,
                        inverse_propensities=inv_ps,
                        seed=seed,
                        return_meta=True,
                        verbose=True,
                        **func[1],
                    )
                    results.update(meta)
                    print(f"  Val size: {val_y_true.shape}")
                    print(f"  Iters: {meta['iters']}")
                    print(f"  Time: {meta['time']:>5.2f} s")
                    save_npz_wrapper(pred_path, y_pred)
                    save_json(results_path, results)
                except Exception as e:
                    raise e
                    print(f"  Failed: {e}")
            else:
                y_pred = load_npz_wrapper(pred_path)
                results = load_json(results_path)
            print(f"  Test size: {y_true.shape}")
            print(f"  Pred size: {y_pred.shape}")
            print("  Metrics (%):")
            results.update(
                calculate_and_report_metrics(
                    y_true, y_pred, k=k, inverse_propensities=inv_ps
                )
            )
            save_json(results_path, results)

        print("  Done")


if __name__ == "__main__":
    main()
