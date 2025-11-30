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
from xcolumns.metrics import *
from xcolumns.weighted_prediction import *


def mean_positive_labels(y_true, y_pred):
    """
    Count the number of positive labels in the true labels.
    """
    return y_true.sum(axis=1).mean()


METRICS_ON_Y = {
    # These are slow to compute (old implementations)
    # "mC": macro_abandonment_v0,
    # "iC": instance_abandonment_v0,
    # "mP": macro_precision_v0,
    # "iP": instance_precision_v0,
    # "mR": macro_recall_v0,
    # "iR": instance_recall_v0,
    # "mF": macro_f1_v0,
    # "iF": instance_f1_v0,
    # "iBA": instance_balanced_accuracy_v0,
    # "mBA": macro_balanced_accuracy_v0,
    "instance-recall": instance_recall,
    "instance-precision": instance_precision,
    "instance-f1": instance_f1_score,
    "instance-balanced-accuracy": instance_balanced_accuracy,
    "abandonment": abandonment,
    "mean_positive_labels": mean_positive_labels,
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
    # conf_mat = calculate_confusion_matrix(y_true, y_pred, normalize=False)
    # print(f"    Confusion matrix: {conf_mat.tp[0]}, {conf_mat.tn[0]}, {conf_mat.fp[0]}, {conf_mat.fn[0]}")

    for metric, func in METRICS_ON_C_MAT.items():
        value = call_function_with_supported_kwargs(
            func, *conf_mat, k=k, w=inverse_propensities, epsilon=1e-9
        )
        results[f"{metric}@{k}"] = value
        print(f"    {metric}@{k}: {100 * value:>5.3f}")

    return results


TOL = 1e-7
METHODS = {
    # Instance-wise measures / baselines
    "optimal-instance-precision": (predict_optimizing_instance_precision, {}),
    # "optimal-instance-precision-keep-scores": (predict_optimizing_instance_precision, {"keep_scores": True}),
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
    "power-law-with-beta=0.5-eps=1e-08": (
        predict_power_law_weighted_per_instance,
        {"beta": 0.5, "epsilon": 1e-8},
    ),
    "power-law-with-beta=0.25-eps=1e-08": (
        predict_power_law_weighted_per_instance,
        {"beta": 0.25, "epsilon": 1e-8},
    ),
    "log-eps=1e-08": (predict_log_weighted_per_instance, {"epsilon": 1e-8}),
    "optimal-macro-recall-eps=1e-08": (
        predict_optimizing_macro_recall,
        {"epsilon": 1e-8},
    ),
    "optimal-macro-balanced-accuracy-eps=1e-08": (
        predict_optimizing_macro_balanced_accuracy,
        {"epsilon": 1e-8},
    ),
    f"block-coord-coverage-tol={TOL}": (
        predict_optimizing_coverage_using_bc,
        {"tolerance": TOL, "max_iters": 100},
    ),
}

# Block Coordinate
for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
    bc_args = {"max_iters": 100, "tolerance": TOL, "metric_kwargs": {"epsilon": eps}}
    greedy_args = {
        "init_y_pred": "greedy",
        "max_iters": 1,
        "metric_kwargs": {"epsilon": eps},
    }
    fw_args = {"max_iters": 100, "metric_kwargs": {"epsilon": eps}}
    fw_1_iter = {"max_iters": 1, "metric_kwargs": {"epsilon": eps}}

    METHODS.update(
        {
            f"block-coord-macro-precision-tol={TOL}-eps={eps}": (
                predict_optimizing_macro_precision_using_bc,
                bc_args,
            ),
            f"block-coord-macro-f1-tol={TOL}-eps={eps}": (
                predict_optimizing_macro_f1_score_using_bc,
                bc_args,
            ),
            f"block-coord-macro-jaccard-score-tol={TOL}-eps={eps}": (
                predict_optimizing_macro_jaccard_score_using_bc,
                bc_args,
            ),
        }
    )

# Frank Wolfe
for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    bc_args = {"max_iters": 100, "tolerance": TOL, "metric_kwargs": {"epsilon": eps}}
    greedy_args = {
        "init_y_pred": "greedy",
        "max_iters": 1,
        "metric_kwargs": {"epsilon": eps},
    }
    fw_args = {"max_iters": 100, "metric_kwargs": {"epsilon": eps}}
    fw_1_iter = {"max_iters": 1, "metric_kwargs": {"epsilon": eps}}

    METHODS.update(
        {
            f"frank-wolfe-macro-precision-eps={eps}": (
                find_and_predict_for_macro_precision_using_fw,
                fw_args,
            ),
            f"frank-wolfe-macro-f1-eps={eps}": (
                find_and_predict_for_macro_f1_score_using_fw,
                fw_args,
            ),
            f"frank-wolfe-macro-jaccard-score-eps={eps}": (
                find_and_predict_for_macro_jaccard_score_using_fw,
                fw_args,
            ),
            # Frank Wolfe with 1 iter
            # f"frank-wolfe-macro-precision-max_iters=1-eps={eps}": (
            #     find_and_predict_for_macro_precision_using_fw,
            #     fw_1_iter,
            # ),
            # f"frank-wolfe-macro-f1-max_iters=1-eps={eps}": (
            #     find_and_predict_for_macro_f1_score_using_fw,
            #     fw_1_iter,
            # ),
            # f"frank-wolfe-macro-jaccard-score-max_iters=1-eps={eps}": (
            #     find_and_predict_for_macro_jaccard_score_using_fw,
            #     fw_1_iter,
            # ),
        }
    )

# Greedy variants
for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2]:
    bc_args = {"max_iters": 100, "tolerance": TOL, "metric_kwargs": {"epsilon": eps}}
    greedy_args = {
        "init_y_pred": "greedy",
        "max_iters": 1,
        "metric_kwargs": {"epsilon": eps},
    }
    fw_args = {"max_iters": 100, "metric_kwargs": {"epsilon": eps}}
    fw_1_iter = {"max_iters": 1, "metric_kwargs": {"epsilon": eps}}

    METHODS.update(
        {
            # Block coordinate
            f"block-coord-macro-recall-tol={TOL}-eps={eps}": (
                predict_optimizing_macro_recall_using_bc,
                bc_args,
            ),
            f"block-coord-macro-balanced-accuracy-tol={TOL}-eps={eps}": (
                predict_optimizing_macro_balanced_accuracy_using_bc,
                bc_args,
            ),
            # Frank Wolfe,
            f"frank-wolfe-macro-recall-eps={eps}": (
                find_and_predict_for_macro_recall_using_fw,
                fw_args,
            ),
            f"frank-wolfe-macro-balanced-accuracy-eps={eps}": (
                find_and_predict_for_macro_balanced_accuracy_using_fw,
                fw_args,
            ),
        }
    )


# Add variants with different alpha for mixed utilities
alphas = [
    # 0.01,
    0.05,
    0.1,
    # 0.2,
    0.3,
    # 0.4,
    0.5,
    # 0.6,
    0.7,
    # 0.8,
    0.9,
    0.95,
    # 0.99,
]
# alphas = [0.95,]
alphas = [0.95, 0.96, 0.99]
for alpha in alphas:
    # METHODS[f"power-law-with-beta={alpha}"] = (
    #     predict_power_law_weighted_per_instance,
    #     {"beta": alpha},
    # )
    METHODS[f"block-coord-mixed-precision-macro-coverage-alpha={alpha}-tol={TOL}"] = (
        predict_optimizing_coverage_using_bc,
        {"alpha": alpha, "tolerance": TOL, "max_iters": 100},
    )

    for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2]:
        bc_args = {
            "max_iters": 100,
            "alpha": alpha,
            "tolerance": TOL,
            "metric_kwargs": {"epsilon": eps},
        }
        fw_args = {"alpha": alpha, "max_iters": 100, "metric_kwargs": {"epsilon": eps}}
        METHODS.update(
            {
                f"block-coord-mixed-precision-macro-f1-alpha={alpha}-tol={TOL}-eps={eps}": (
                    predict_optimizing_mixed_instance_precision_and_macro_f1_score_using_bc,
                    bc_args,
                ),
                # f"block-coord-mixed-precision-macro-precision-alpha={alpha}-tol={TOL}-eps={eps}": (
                #     predict_optimizing_mixed_instance_precision_and_macro_precision_using_bc,
                #     bc_args,
                # ),
                f"block-coord-mixed-precision-macro-recall-alpha={alpha}-tol={TOL}-eps={eps}": (
                    predict_optimizing_mixed_instance_precision_and_macro_recall_using_bc,
                    bc_args,
                ),
                f"frank-wolfe-mixed-precision-macro-recall-alpha={alpha}-eps={eps}": (
                    find_and_predict_for_mixed_instance_precision_and_macro_recall_using_fw,
                    fw_args,
                ),
            }
        )

    for eps in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2]:
        fw_args = {"alpha": alpha, "max_iters": 100, "metric_kwargs": {"epsilon": eps}}
        METHODS.update(
            {
                f"frank-wolfe-mixed-precision-macro-f1-alpha={alpha}-eps={eps}": (
                    find_and_predict_for_mixed_instance_precision_and_macro_f1_score_using_fw,
                    fw_args,
                ),
                # f"frank-wolfe-mixed-precision-macro-precision-alpha={alpha}-eps={eps}": (
                #     find_and_predict_for_mixed_instance_precision_and_macro_precision_using_fw,
                #     fw_args,
                # ),
            }
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
@click.option(
    "--only_recalculate_results", is_flag=True, type=bool, required=False, default=False
)
@click.option("--shuffle_data", is_flag=True, type=bool, required=False, default=False)
@click.option("--multiply_data", type=int, required=False, default=1)
@click.option("--multiply_test", type=int, required=False, default=1)
@click.option("--use_proba_as_true", is_flag=True, required=False, default=False)
@click.option("--use_proba_as_val", is_flag=True, required=False, default=False)
@click.option(
    "--use_proba_mul_true_as_val", is_flag=True, required=False, default=False
)
@click.option("--use_train_as_test", is_flag=True, required=False, default=False)
@click.option("--use_dense", is_flag=True, type=bool, required=False, default=False)
@click.option("-v", "--val_split", type=float, required=False, default=0.0)
@click.option("-t", "--proba_threshold", type=float, required=False, default=0.0)
@click.option("-i", "--max_iters", type=int, required=False, default=100)
@click.option(
    "--sample_test_labels", is_flag=True, type=bool, required=False, default=False
)
@click.option("--sample_test_seed", type=int)
@click.option("--top_labels", type=float, default=1.0)
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
    only_recalculate_results,
    shuffle_data,
    multiply_data,
    multiply_test,
    use_proba_as_true,
    use_proba_as_val,
    use_proba_mul_true_as_val,
    use_train_as_test,
    use_dense,
    val_split,
    proba_threshold,
    max_iters,
    sample_test_labels,
    sample_test_seed,
    top_labels,
):
    if method is None:
        methods = METHODS
    elif method in METHODS:
        methods = {method: METHODS[method]}
    else:
        _method = method.split(",")
        print(f"Method: {_method}")
        methods = METHODS
        for m in _method:
            methods = {k: v for k, v in methods.items() if m in k}
            # print(f"Methods: {methods.keys()}")
    if len(methods) == 0:
        raise ValueError(f"Unknown method: {method}")

    use_true_as_pred = "use_true_as_pred" in experiment
    lightxml_data = "lightxml" in experiment
    inv_ps_a = 0.55
    inv_ps_b = 1.5

    if "wikiLSHTC_" in experiment or "WikipediaLarge-500K_" in experiment:
        inv_ps_a = 0.5
        inv_ps_b = 0.4
    elif "amazon_" in experiment or "amazon-3M_" in experiment:
        inv_ps_a = 0.6
        inv_ps_b = 2.6

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
        "EURLex-4.3K",
        "amazon-3M",
        "amazonCat-14K",
        "rcv1x",
        "eurlex",
        "amazonCat",
        "wiki10",
        "amazon",
        "wikiLSHTC",
        "WikipediaLarge-500K",
        "deliciousLarge",
    ]
    for d in nxc_datasets:
        if d in experiment:
            _d = d
            if "l2" in experiment:
                _d += "_l2"
            y_proba_path = {
                "path": f"predictions/nxc/{_d}/test_pred",
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
            if os.path.exists(f"predictions/nxc/{_d}/train_pred"):
                train_y_proba_path = {
                    "path": f"predictions/nxc/{_d}/train_pred",
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

    print(f"y_true: type={type(y_true)}, shape={y_true.shape}")
    print(f"y_proba: type={type(y_proba)}, shape={y_proba.shape}")

    if train_y_true is not None and train_y_proba is not None:
        if train_y_true.shape != y_true.shape:
            align_dim1(y_true, train_y_true)

        if train_y_proba.shape != y_true.shape:
            align_dim1(y_true, train_y_proba)

    # For some sparse format this resize might be necessary
    if y_true.shape != y_proba.shape:
        if y_true.shape[0] != y_proba.shape[0]:
            raise RuntimeError(
                f"Number of instances in true and prediction do not match {y_true.shape[0]} != {y_proba.shape[0]}"
            )
        align_dim1(y_true, y_proba)

    if "apply_sigmoid" in y_proba_path and y_proba_path["apply_sigmoid"]:
        # LightXML predictions aren't probabilities
        y_proba.data = 1.0 / (1.0 + np.exp(-y_proba.data))

    if shuffle_data and train_y_true is not None and train_y_proba is not None:
        all_y_true = sp.vstack([train_y_true, y_true])
        all_y_proba = sp.vstack([train_y_proba, y_proba])

    if multiply_data > 1:
        all_y_true = sp.vstack([all_y_true] * multiply_data)
        all_y_proba = sp.vstack([all_y_proba] * multiply_data)

    if shuffle_data:
        y_true, train_y_true, y_proba, train_y_proba = train_test_split(
            all_y_true, all_y_proba, test_size=0.5, random_state=seed
        )

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
        print(f"Calculating priors and propensities A={inv_ps_a}, B={inv_ps_b}")
        if val_y_true is not None:
            priors = labels_priors(val_y_true)
        else:
            priors = labels_priors(train_y_true)
        inv_ps = jpv_inverse_propensity(train_y_true, A=inv_ps_a, B=inv_ps_b)

    def zero_not_in_the_list(matrix: csr_matrix, list: np.array):
        for i in tqdm(range(matrix.shape[0])):
            row = matrix[i]
            mask = np.isin(row.indices, list, assume_unique=True)
            # row.data[~mask] = 0
            matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]][~mask] = 0

    # Release some memory
    if only_recalculate_results:
        print("Releasing memory")
        del train_y_proba
        del train_y_true
        del val_y_proba
        del val_y_true
        train_y_proba = None
        train_y_true = None
        val_y_proba = None
        val_y_true = None

    # Sample test labels based on the provided marginals
    if sample_test_labels:
        np.random.seed(sample_test_seed)
        print("Sampling labels using seed:", sample_test_seed)
        y_true = y_proba.copy()
        # print(y_proba[0])
        y_true.data = (y_true.data >= np.random.random(y_true.data.shape[0])).astype(
            np.float64
        )
        y_true.eliminate_zeros()
        # print(y_true[0])
        # print(y_true.sum(axis=0).shape, y_true.sum(axis=0)[0])
        # print(y_proba[0])

    if top_labels < 1.0:
        print(y_proba[0].data.shape)
        n_top_labels = int(top_labels * priors.shape[0])
        labels_to_keep = np.argpartition(-priors, n_top_labels)[:n_top_labels]
        print(labels_to_keep, labels_to_keep.shape)
        zero_not_in_the_list(y_proba, labels_to_keep)
        y_proba.eliminate_zeros()
        print("Eliminated probas:", y_proba[0].data.shape)
        # if train_y_proba is not None:
        #     zero_not_in_the_list(train_y_proba, labels_to_keep)
        #     train_y_proba.eliminate_zeros()
        # if val_y_proba is not None and train_y_proba.shape != val_y_proba.shape:
        #     zero_not_in_the_list(val_y_proba, labels_to_keep)
        #     val_y_proba.eliminate_zeros()

    # Use true labels as predictions with 1.0 score (probability)
    if multiply_test > 1:
        y_true = sp.vstack([y_true] * multiply_test)
        y_proba = sp.vstack([y_proba] * multiply_test)

    if use_true_as_pred:
        y_proba = y_true

    if use_proba_as_true:
        y_true = y_proba
        if val_y_proba is not None and val_y_true is not None:
            val_y_true = val_y_proba
        if train_y_proba is not None and train_y_true is not None:
            train_y_true = train_y_proba

    if use_proba_as_val:
        val_y_true = val_y_proba

    if use_proba_mul_true_as_val:
        val_y_true = val_y_true.multiply(val_y_proba)

    if use_train_as_test:
        y_true = train_y_true
        y_proba = train_y_proba
        if val_y_proba is not None and val_y_true is not None:
            val_y_true = train_y_true
            val_y_proba = train_y_proba

    print(f"y_true: type={type(y_true)}, shape={y_true.shape}")
    print(f"y_proba: type={type(y_proba)}, shape={y_proba.shape}")
    if train_y_true is not None:
        print(f"train_y_true: type={type(train_y_true)}, shape={train_y_true.shape}")

    if proba_threshold > 0:

        def zero_below_threshold_and_keep_top_k(matrix: csr_matrix, threshold, k):
            for i in tqdm(range(matrix.shape[0])):
                row = matrix[i]
                indices = np.argsort(row.data)[-k:]
                mask = np.zeros_like(row.data, dtype=bool)
                mask[row.data < threshold] = True
                mask[indices] = False
                matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]][mask] = 0

        zero_below_threshold_and_keep_top_k(y_proba, proba_threshold, k)
        y_proba.eliminate_zeros()
        print("Eliminated probas:", y_proba[0].data.shape)
        if train_y_proba is not None:
            zero_below_threshold_and_keep_top_k(train_y_proba, proba_threshold, k)
            train_y_proba.eliminate_zeros()
        if val_y_proba is not None and train_y_proba.shape != val_y_proba.shape:
            zero_below_threshold_and_keep_top_k(val_y_proba, proba_threshold, k)
            val_y_proba.eliminate_zeros()

        # y_proba.data[y_proba.data < proba_threshold] = 0
        # val_y_proba.data[val_y_proba.data < proba_threshold] = 0
        # train_y_proba.data[train_y_proba.data < proba_threshold] = 0

    # Convert to array to check if it gives the same results
    if use_dense:
        y_true = y_true.toarray()
        y_proba = y_proba.toarray()
        train_y_true = train_y_true.toarray()

    # print(y_pred.shape, inv_ps.shape, priors.shape, y_true.shape, y_true_test.shape, y_proba.shape, y_proba_test.shape)

    output_path_prefix = f"{results_dir}/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for method, func in methods.items():
        print(f"{experiment} - {method} @ {k} ({seed=}, kwargs={func[1]}): ")

        if top_labels < 1.0:
            method += f"_top_labels={top_labels}"
        if proba_threshold > 0:
            method += f"_proba_threshold={proba_threshold}"
        if use_proba_as_val:
            method += "_proba_as_val"
        if use_proba_mul_true_as_val:
            method += "_proba_mul_true_as_val"
        if multiply_test > 1:
            method += f"_multiply_test={multiply_test}"

        output_path = f"{output_path_prefix}{method}_k={k}_v={val_split}_s={seed}"
        results_path = f"{output_path}_results.json"
        pred_path = f"{output_path}_pred.npz"

        if (
            not os.path.exists(results_path)
            or recalculate_results
            or recalculate_predictions
            or only_recalculate_results
        ):
            results = {}
            if (
                not os.path.exists(pred_path) or recalculate_predictions
            ) and not only_recalculate_results:
                try:
                    if "max_iters" in func[1]:
                        func[1]["max_iters"] = max_iters

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
                    if val_y_true is not None:
                        print(f"  Val shape: {val_y_true.shape}")
                    print(f"  Iters: {meta.get('iters', 1)}")
                    print(f"  Time: {meta['time']:>5.2f} s")
                    save_npz_wrapper(pred_path, y_pred)
                    save_json(results_path, results)
                except Exception as e:
                    raise e
                    # print(f"  Failed: {e}")
            elif os.path.exists(pred_path):
                y_pred = load_npz_wrapper(pred_path)
                results = load_json(results_path)
            else:
                continue
            print(y_true.sum(axis=0).shape, y_true.sum(axis=0)[0])
            print(f"  Test shape: {y_true.shape}")
            print(f"  Pred shape: {y_pred.shape}")
            print("  Metrics (%):")
            results["pred_shape"] = y_pred.shape
            results.update(
                calculate_and_report_metrics(
                    y_true, y_pred, k=k, inverse_propensities=inv_ps
                )
            )

            if sample_test_labels:
                results_path = f"{output_path_prefix}{method}_sample_test_labels_k={k}_v={val_split}_s={seed}_sample_s={sample_test_seed}_results.json"

            save_json(results_path, results)

        print("  Done")


if __name__ == "__main__":
    main()
