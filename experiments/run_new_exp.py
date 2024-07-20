import sys

import click
import numpy as np
import scipy.sparse as sp
import torch
from napkinxc.datasets import load_libsvm_file, to_csr_matrix
from napkinxc.models import BR, PLT
from pytorch_models.baseline_classifiers import *
from pytorch_models.losses import *
from pytorch_models.transformer_classifier import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from skmultilearn.problem_transform import LabelPowerset
from tqdm import trange

from utils import *
from xcolumns.block_coordinate import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics import *
from xcolumns.metrics_original import *
from xcolumns.types import *
from xcolumns.weighted_prediction import *


def frank_wolfe_wrapper(
    Y_val,
    pred_val,
    pred_test,
    utility_func,
    k: int = 5,
    seed: int = 0,
    pred_repeat=10,
    pred_strategy="default",  # default, last, average
    average_tail_min_count=10,
    **kwargs,
):
    rnd_classifier, meta = call_function_with_supported_kwargs(
        find_classifier_using_fw,
        Y_val,
        pred_val,
        utility_func,
        max_iters=50,
        k=k,
        seed=seed,
        return_meta=True,
        verbose=True,
        **kwargs,
    )
    print(f"  classifiers weights (probabilities): {rnd_classifier.p}")
    y_preds = []
    if pred_strategy == "default":
        print("  predicting with randomized classfier")
        for i in range(pred_repeat):
            y_pred = rnd_classifier.predict(pred_test, seed=seed + i)
            y_preds.append(y_pred)
    elif pred_strategy == "last":
        print("  using last classifier")
        y_pred = predict_using_randomized_weighted_classifier(
            pred_test,
            rnd_classifier.a[-1:],
            rnd_classifier.a[-1:],
            np.array([1]),
            k=rnd_classifier.k,
            seed=seed,
        )
    elif pred_strategy in ["average" or "average_tail"]:
        print("  averaging classifiers weights")
        avg_a = np.zeros((rnd_classifier.a.shape[1], rnd_classifier.a.shape[2]))
        avg_b = np.zeros((rnd_classifier.b.shape[1], rnd_classifier.b.shape[2]))
        for i in range(rnd_classifier.p.shape[0]):
            avg_a += rnd_classifier.p[i] * rnd_classifier.a[i]
            avg_b += rnd_classifier.p[i] * rnd_classifier.b[i]
        if pred_strategy == "average_tail":
            pass
        y_pred = predict_weighted_per_instance(pred_test, a=avg_a, b=avg_b, k=k)
        y_preds.append(y_pred)

    return y_preds, meta


def new_learning_wrapper(
    Y_val,
    pred_val,
    pred_test,
    utility_func,
    k: int = 5,
    seed: int = 0,
    pred_repeat=10,
    data_repeat=10,
    pred_strategy="default",  # default, last, average
    average_tail_min_count=10,
    X_val=None,
    X_test=None,
    model_path=None,
    **kwargs,
):
    Y_val = sp.vstack([Y_val for _ in range(data_repeat)])
    X_val = sp.vstack([X_val for _ in range(data_repeat)])
    pred_val = sp.vstack([pred_val for _ in range(data_repeat)])
    # Y_val = np.tile(Y_val, (pred_repeat, 1))
    # X_val = np.tile(X_val, (pred_repeat, 1))
    Y_new_labels, meta = predict_using_bc_with_0approx(
        pred_val,
        utility_func,
        k,
        seed=seed,
        return_meta=True,
        verbose=True,
    )

    print(Y_new_labels.shape)
    print(Y_new_labels.indptr, Y_new_labels.data, Y_new_labels.indices)
    print(Y_new_labels.sum(axis=1))

    model_seed = 13
    model = NapkinModel(
        model_path + "_model",
        model_seed,
        PLT(
            model_path,
            verbose=True,
            threads=15,
            seed=model_seed,
            max_leaves=200,
            # liblinear_eps=0.001,
            # liblinear_c=16,
            liblinear_eps=0.01,
            liblinear_c=16,
            # ensemble=3,
        ),
    )
    model.fit(X_val, Y_new_labels)
    y_proba = model.predict_proba(X_test, 200)
    print(X_test.shape, pred_test.shape, y_proba.shape)
    y_preds = medow_sampling_csr(y_proba, k)
    print("====================")
    print(y_proba[0])
    print("--------------------")
    print(y_preds[0])
    print("--------------------")
    print(pred_test[0])
    print("--------------------")
    print(Y_new_labels[0])
    print("====================")
    y_preds = [y_preds]

    return y_preds, meta


def medow_sampling_csr(y_proba, k):
    n = y_proba.shape[0]
    y_pred_data = np.ones(n * k, dtype=DefaultDataDType)
    y_pred_indices = np.zeros(n * k, dtype=DefaultIndDType)
    y_pred_indptr = np.zeros(n + 1, dtype=DefaultIndDType)

    for i in range(y_proba.shape[0]):
        u = np.random.rand()
        s_i = 0
        y_i = 0
        while s_i < k and y_proba.indptr[i] + y_i < y_proba.indptr[i + 1]:
            u -= y_proba.data[y_proba.indptr[i] + y_i]
            if u + s_i <= 0:
                y_pred_indices[i * k + s_i] = y_proba.indices[y_pred_indptr[i] + y_i]
                s_i += 1
            y_i += 1
        y_pred_indptr[i + 1] = i * k

    print(y_pred_data.shape, y_pred_indices.shape, y_pred_indptr.shape)
    return sp.csr_matrix(
        (y_pred_data, y_pred_indices, y_pred_indptr), shape=(n, y_proba.shape[1])
    )


def frank_wolfe_macro_recall(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return frank_wolfe_wrapper(
        Y_val,
        pred_val,
        pred_test,
        macro_recall_on_conf_matrix,
        k=k,
        seed=seed,
        **kwargs,
    )


def frank_wolfe_macro_precision(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return frank_wolfe_wrapper(
        Y_val,
        pred_val,
        pred_test,
        macro_precision_on_conf_matrix,
        k=k,
        seed=seed,
        **kwargs,
    )


def frank_wolfe_macro_f1(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return frank_wolfe_wrapper(
        Y_val,
        pred_val,
        pred_test,
        macro_f1_score_on_conf_matrix,
        k=k,
        seed=seed,
        **kwargs,
    )


def frank_wolfe_balanced_accuracy(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return frank_wolfe_wrapper(
        Y_val,
        pred_val,
        pred_test,
        macro_balanced_accuracy_on_conf_matrix,
        k=k,
        seed=seed,
        **kwargs,
    )


def frank_wolfe_mixed_instance_prec_macro_f1(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, alpha=0, **kwargs
):
    def mixed_utility_fn(tp, fp, fn, tn):
        return (1 - alpha) * precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * macro_f1_score_on_conf_matrix(tp, fp, fn, tn)

    return frank_wolfe_wrapper(
        Y_val, pred_val, pred_test, mixed_utility_fn, k=k, seed=seed, **kwargs
    )


def frank_wolfe_mixed_instance_prec_macro_prec(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, alpha=0, **kwargs
):
    def mixed_utility_fn(tp, fp, fn, tn):
        return (1 - alpha) * precision_at_k_on_conf_matrix(
            tp, fp, fn, tn, k
        ) + alpha * macro_precision_on_conf_matrix(tp, fp, fn, tn)

    return frank_wolfe_wrapper(
        Y_val, pred_val, pred_test, mixed_utility_fn, k=k, seed=seed, **kwargs
    )


def report_metrics(data, predictions, k):
    results = {}
    for metric, func in METRICS.items():
        values = []
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]
        for pred in predictions:
            value = func(data, pred)
            values.append(value)
        results[f"{metric}@{k}"] = values
        # print(values)
        print(
            f"  {metric}: {100 * np.mean(values):>5.2f} +/- {100 * np.std(values):>5.2f}"
        )

    return results


def fw_optimal_instance_precision_wrapper(
    Y_val, pred_val, pred_test, k: int = 5, **kwargs
):
    return call_function_with_supported_kwargs(
        predict_optimizing_instance_precision, pred_test, k, return_meta=True, **kwargs
    )


def fw_optimal_macro_recall_wrapper(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, priors=None, **kwargs
):
    return call_function_with_supported_kwargs(
        predict_optimizing_macro_recall,
        pred_test,
        k,
        priors=priors,
        return_meta=True,
        **kwargs,
    )


def fw_power_law_weighted_instance_wrapper(
    Y_val,
    pred_val,
    pred_test,
    k: int = 5,
    seed: int = 0,
    priors=None,
    beta=0.5,
    **kwargs,
):
    return call_function_with_supported_kwargs(
        predict_power_law_weighted_per_instance,
        pred_test,
        k,
        priors=priors,
        beta=beta,
        return_meta=True,
        **kwargs,
    )


def fw_log_weighted_instance_wrapper(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, priors=None, **kwargs
):
    return call_function_with_supported_kwargs(
        predict_log_weighted_per_instance,
        pred_test,
        k,
        priors=priors,
        return_meta=True,
        **kwargs,
    )


def fw_optimal_macro_balanced_accuracy_wrapper(
    Y_val,
    pred_val,
    pred_test,
    k: int = 5,
    seed: int = 0,
    priors=None,
    return_meta=True,
    **kwargs,
):
    return call_function_with_supported_kwargs(
        predict_optimizing_macro_balanced_accuracy,
        pred_test,
        k,
        priors=priors,
        return_meta=True,
        **kwargs,
    )


def fw_bca_macro_precision_wrapper(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return block_coordinate_macro_precision(pred_test, k=k, **kwargs)


def fw_bca_macro_f1_wrapper(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return block_coordinate_macro_f1(pred_test, k=k, **kwargs)


def fw_bca_macro_recall_wrapper(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return block_coordinate_macro_recall(pred_test, k=k, **kwargs)


def new_learning_wrapper_macro_f1(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return new_learning_wrapper(
        Y_val,
        pred_val,
        pred_test,
        binary_f1_score_on_conf_matrix,
        k=k,
        seed=seed,
        **kwargs,
    )


def new_learning_wrapper_macro_recall(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return new_learning_wrapper(
        Y_val,
        pred_val,
        pred_test,
        binary_recall_on_conf_matrix,
        k=k,
        seed=seed,
        **kwargs,
    )


def new_learning_wrapper_macro_precision(
    Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs
):
    return new_learning_wrapper(
        Y_val,
        pred_val,
        pred_test,
        binary_precision_on_conf_matrix,
        k=k,
        seed=seed,
        **kwargs,
    )


METRICS = {
    "mC": macro_abandonment,
    "iC": instance_abandonment,
    "mP": macro_precision,
    "iP": instance_precision,
    "mR": macro_recall,
    "iR": instance_recall,
    "mF": macro_f1,
    "iF": instance_f1,
    "mBA": macro_balanced_accuracy,
    "iBA": instance_balanced_accuracy,
}


METHODS = {
    "fw-split-optimal-instance-prec": (fw_optimal_instance_precision_wrapper, {}),
    "fw-split-optimal-macro-recall": (fw_optimal_macro_recall_wrapper, {}),
    "fw-split-optimal-macro-balanced-acc": (
        fw_optimal_macro_balanced_accuracy_wrapper,
        {},
    ),
    "fw-split-power-law-with-beta=0.5": (
        fw_power_law_weighted_instance_wrapper,
        {"beta": 0.5},
    ),
    "fw-split-power-law-with-beta=0.25": (
        fw_power_law_weighted_instance_wrapper,
        {"beta": 0.25},
    ),
    "fw-split-log": (fw_log_weighted_instance_wrapper, {}),
    "frank-wolfe-macro-recall": (frank_wolfe_macro_recall, {}),
    "frank-wolfe-macro-prec": (frank_wolfe_macro_precision, {}),
    "frank-wolfe-macro-f1": (frank_wolfe_macro_f1, {}),
    "frank-wolfe-macro-balanced-acc": (frank_wolfe_balanced_accuracy, {}),
    # "fw-split-bca-macro-recall": (fw_bca_macro_recall_wrapper, {}),
    # "fw-split-bca-macro-prec": (fw_bca_macro_precision_wrapper, {}),
    # "fw-split-bca-macro-f1": (fw_bca_macro_f1_wrapper, {}),
    # "frank-wolfe-macro-recall-last": (frank_wolfe_macro_recall, {"use_last": True}),
    # "frank-wolfe-macro-precision-last": (frank_wolfe_macro_precision, {"use_last": True}),
    # "frank-wolfe-macro-f1-last": (frank_wolfe_macro_f1, {"use_last": True}),
    # "frank-wolfe-macro-recall-avg": (frank_wolfe_macro_recall, {"average": True}),
    # "frank-wolfe-macro-precision-avg": (frank_wolfe_macro_precision, {"average": True}),
    # "frank-wolfe-macro-f1-avg": (frank_wolfe_macro_f1, {"average": True}),
    "frank-wolfe-macro-recall-rnd": (
        frank_wolfe_macro_recall,
        {"init_classifier": "random"},
    ),
    "frank-wolfe-macro-prec-rnd": (
        frank_wolfe_macro_precision,
        {"init_classifier": "random"},
    ),
    "frank-wolfe-macro-f1-rnd": (frank_wolfe_macro_f1, {"init_classifier": "random"}),
    "frank-wolfe-macro-balanced-acc-rnd": (
        frank_wolfe_balanced_accuracy,
        {"init_classifier": "random"},
    ),
    "frank-wolfe-macro-recall-topk": (
        frank_wolfe_macro_recall,
        {"init_classifier": "top"},
    ),
    "frank-wolfe-macro-prec-topk": (
        frank_wolfe_macro_precision,
        {"init_classifier": "top"},
    ),
    "frank-wolfe-macro-f1-topk": (frank_wolfe_macro_f1, {"init_classifier": "top"}),
    "frank-wolfe-macro-balanced-acc-topk": (
        frank_wolfe_balanced_accuracy,
        {"init_classifier": "top"},
    ),
    "new-method-macro-f1": (new_learning_wrapper_macro_f1, {}),
    "new-method-macro-recall": (new_learning_wrapper_macro_recall, {}),
    "new-method-macro-precision": (new_learning_wrapper_macro_precision, {}),
    # "frank-wolfe-macro-recall-classic": (frank_wolfe_macro_recall, {"search_for_best_alpha": False}),
    # "frank-wolfe-macro-prec-classic": (frank_wolfe_macro_precision, {"search_for_best_alpha": False}),
    # "frank-wolfe-macro-f1-classic": (frank_wolfe_macro_f1, {"search_for_best_alpha": False}),
    # "frank-wolfe-macro-balanced-acc-classic": (frank_wolfe_balanced_accuracy, {"search_for_best_alpha": False}),
    # "frank-wolfe-macro-prec-step=0.01": (
    #     frank_wolfe_macro_precision,
    #     {"alpha_uniform_search_step": 0.01},
    # ),
    # "frank-wolfe-macro-f1-step=0.01": (
    #     frank_wolfe_macro_f1,
    #     {"alpha_uniform_search_step": 0.01},
    # ),
    # "frank-wolfe-macro-prec-step=0.0001": (
    #     frank_wolfe_macro_precision,
    #     {"alpha_uniform_search_step": 0.0001},
    # ),
    # "frank-wolfe-macro-f1-step=0.0001": (
    #     frank_wolfe_macro_f1,
    #     {"alpha_uniform_search_step": 0.0001},
    # ),
}

# alphas = [
#     # 0.01,
#     #0.05,
#     0.1,
#     #0.2,
#     0.3,
#     #0.4,
#     0.5,
#     #0.6,
#     0.7,
#     #0.8,
#     0.9,
#     #0.95,
#     # 0.99,
#     # 0.995,
#     # 0.999,
# ]
# for alpha in alphas:
#     METHODS[f"frank-wolfe-mixed-prec-f1-alpha={alpha}-step=0.001"] = (
#         frank_wolfe_mixed_instance_prec_macro_f1,
#         {"alpha": alpha, "alpha_uniform_search_step": 0.001},
#     )
#     METHODS[f"frank-wolfe-mixed-prec-prec-alpha={alpha}-step=0.001"] = (
#         frank_wolfe_mixed_instance_prec_macro_prec,
#         {"alpha": alpha, "alpha_uniform_search_step": 0.001},
#     )
#     METHODS[f"frank-wolfe-mixed-prec-f1-alpha={alpha}-step=0.01"] = (
#         frank_wolfe_mixed_instance_prec_macro_f1,
#         {"alpha": alpha, "alpha_uniform_search_step": 0.01},
#     )
#     METHODS[f"frank-wolfe-mixed-prec-prec-alpha={alpha}-step=0.01"] = (
#         frank_wolfe_mixed_instance_prec_macro_prec,
#         {"alpha": alpha, "alpha_uniform_search_step": 0.01},
#     )
#     METHODS[f"frank-wolfe-mixed-prec-f1-alpha={alpha}-step=0.0001"] = (
#         frank_wolfe_mixed_instance_prec_macro_f1,
#         {"alpha": alpha, "alpha_uniform_search_step": 0.0001},
#     )
#     METHODS[f"frank-wolfe-mixed-prec-prec-alpha={alpha}-step=0.0001"] = (
#         frank_wolfe_mixed_instance_prec_macro_prec,
#         {"alpha": alpha, "alpha_uniform_search_step": 0.0001},
#     )
#     METHODS[f"fw-power-law-with-beta={alpha}"] = (
#         fw_power_law_weighted_instance_wrapper,
#         {"beta": alpha},
#     )


def log_loss(true, pred):
    true = true.toarray()
    pred = pred.toarray()
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))


def load_txt_data():
    pass


class ModelWrapper:
    def __init__(self, model_path, seed):
        self.model_path = model_path
        self.seed = seed
        self.model = None

    def fit(self, X, Y):
        pass

    def predict_proba(self, X, top_k):
        pass


class NapkinModel(ModelWrapper):
    def __init__(self, model_path, seed, napkin_model):
        super().__init__(model_path, seed)
        self.model = napkin_model

    def fit(self, X, Y, *args, **kwargs):
        self.model.fit(X, Y)

    def predict_proba(self, X, top_k):
        pred = self.model.predict_proba(X, top_k=top_k)
        pred = to_csr_matrix(pred, sort_indices=True)
        return pred


class StackedNapkinModel(ModelWrapper):
    def __init__(self, model_path, seed, first_napkin_model, second_napkin_model):
        super().__init__(model_path, seed)
        self.first_model = first_napkin_model
        self.second_model = second_napkin_model
        self.first_top_k = 10

    def _create_data_for_second_model(self, X, first_Y):
        second_X = sp.hstack((X, first_Y))
        return second_X

    def fit(self, X, Y, *args, **kwargs):
        print("  Training first model ...")
        X = normalize(X, norm="l2")
        self.first_model.fit(X, Y)
        first_pred_Y = self.first_model.predict_proba(X, top_k=self.first_top_k)
        first_pred_Y = to_csr_matrix(first_pred_Y, sort_indices=True)

        second_X = self._create_data_for_second_model(X, first_pred_Y)

        print("  Training second model ...")
        self.second_model.fit(second_X, Y)

    def predict_proba(self, X, top_k):
        X = normalize(X, norm="l2")
        first_pred_Y = self.first_model.predict_proba(X, top_k=self.first_top_k)
        first_pred_Y = to_csr_matrix(first_pred_Y, sort_indices=True)

        second_X = self._create_data_for_second_model(X, first_pred_Y)
        pred = self.second_model.predict_proba(second_X, top_k=top_k)
        pred = to_csr_matrix(pred, sort_indices=True)
        return pred


class SklearnModel(ModelWrapper):
    def __init__(self, model_path, seed, model):
        super().__init__(model_path, seed)
        self.model = model

    def fit(self, X, Y, *args, **kwargs):
        self.model.fit(X, Y)

    def predict_proba(self, X, top_k):
        pred = self.model.predict_proba(X)
        pred = to_csr_matrix(pred, sort_indices=True)
        return pred


lr = 0.001  # 0.01 - eurlecx and wiki10, 0.02 - amazonCat and amazon-670k
wd = 1e-4
max_epochs = 1
adam_eps = 1e-7
train_batch_size = 64
eval_batch_size = 8 * train_batch_size
num_workers = 8
precision = 16

torch.set_float32_matmul_precision("medium")


class PytorchModel(ModelWrapper):
    def __init__(
        self, model_path, seed, loss="bce", hidden_units=()
    ):  # 512 for mediamill, 1024 for flicker, 2048 for rcv1x
        super().__init__(model_path, seed)

        self.loss = None
        if loss == "bce":
            self.loss = F.binary_cross_entropy_with_logits
        elif loss == "focal":
            self.loss = FocalLoss()
        elif loss == "asym":
            self.loss = AsymmetricLoss()

        print("Loss:", self.loss.__class__.__name__)
        # self.model = TransformerClassfier("destil-bert",
        #                                   self.loss,
        #                                   learning_rate=1e-5,  # początkowy rozmiar kroku uczenia
        #                                   weight_decay = 0.00001,
        #                                   max_epochs=1,
        #                                   precision=16,
        #                                   ckpt_dir=model_path)
        self.model = FlatFullyConnectedClassfier(
            self.loss,
            learning_rate=0.01,
            weight_decay=0.00001,
            max_epochs=3,
            precision=16,
            hidden_units=hidden_units,
            ckpt_dir=model_path,
        )

    def fit(self, X, Y, X_val, Y_val):
        # self.model.fit(X, Y, X_val=X_val, Y_val=Y_val)
        self.model.fit(X, Y)

    def predict_proba(self, X, top_k):
        pred = []
        batch_size = 64 * 1024
        rows = 0
        print("Predicting ...")
        while rows < X.shape[0]:
            _pred = self.model.predict(X[rows : min(rows + batch_size, X.shape[0])])
            _pred = torch.vstack(_pred)
            pred.append(_pred.cpu().numpy())
            rows += batch_size

        pred = np.vstack(pred)
        print("Converting to csr matrix ...")
        shape = pred.shape
        if shape[1] > top_k:
            size = shape[0] * top_k
            indptr = np.zeros(shape[0] + 1, dtype=np.int32)
            indices = np.zeros(size, dtype=np.int32)
            data = np.ones(size, dtype=np.float32)
            cells = 0
            indptr[0] = 0

            for i in trange(shape[0]):
                top_k_indicies = np.argpartition(-pred[i], top_k)[:top_k]
                indptr[i + 1] = indptr[i] + top_k
                indices[cells : cells + top_k] = top_k_indicies
                data[cells : cells + top_k] = pred[i][top_k_indicies]
                cells += top_k

            pred = sp.csr_matrix((data, indices, indptr), shape=shape)
        else:
            pred = sp.csr_matrix(pred)
        return pred


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("-k", type=int, required=True)
@click.option("-s", "--seed", type=int, required=True)
@click.option("-m", "--method", type=str, required=False, default=None)
@click.option("-t", "--testsplit", type=float, required=False, default=0)
@click.option("-r", "--results_dir", type=str, required=False, default="results_fw/")
@click.option(
    "--recalculate_predictions", is_flag=True, type=bool, required=False, default=False
)
@click.option(
    "--recalculate_results", is_flag=True, type=bool, required=False, default=False
)
@click.option("--retrain_model", is_flag=True, type=bool, required=False, default=False)
def main(
    experiment,
    k,
    seed,
    method,
    testsplit,
    results_dir,
    recalculate_predictions,
    recalculate_results,
    retrain_model,
):
    print(experiment)

    if method is None:
        methods = METHODS
    elif method in METHODS:
        methods = {method: METHODS[method]}
    else:
        raise ValueError(f"Unknown method: {method}")

    data_load_config = {
        "labels_delimiter": ",",
        "labels_features_delimiter": " ",
        "header": True,
    }

    if "yeast" in experiment:
        data_load_config["header"] = False
        test_path = {
            "path": "datasets/yeast/yeast_test.txt",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/yeast/yeast_train.txt",
            "load_func": load_txt_data,
        }

    elif "youtube_deepwalk" in experiment:
        data_load_config["header"] = False
        test_path = {
            "path": "datasets/youtube_deepwalk/youtube_deepwalk_test.svm",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/youtube_deepwalk/youtube_deepwalk_train.svm",
            "load_func": load_txt_data,
        }

    elif "eurlex_lexglue" in experiment:
        data_load_config["header"] = False
        test_path = {
            "path": "datasets/eurlex_lexglue/eurlex_lexglue_test.svm",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/eurlex_lexglue/eurlex_lexglue_train.svm",
            "load_func": load_txt_data,
        }

    elif "mediamill" in experiment:
        data_load_config["header"] = False
        test_path = {
            "path": "datasets/mediamill/mediamill_test.txt",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/mediamill/mediamill_train.txt",
            "load_func": load_txt_data,
        }

    elif "bibtex" in experiment:
        data_load_config["header"] = False
        test_path = {
            "path": "datasets/bibtex/bibtex_test.svm",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/bibtex/bibtex_train.svm",
            "load_func": load_txt_data,
        }

    elif "delicious" in experiment:
        data_load_config["header"] = False
        test_path = {
            "path": "datasets/delicious/delicious_test.svm",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/delicious/delicious_train.svm",
            "load_func": load_txt_data,
        }

    elif "flicker_deepwalk" in experiment:
        data_load_config["header"] = False
        test_path = {
            "path": "datasets/flicker_deepwalk/flicker_deepwalk_test.svm",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/flicker_deepwalk/flicker_deepwalk_train.svm",
            "load_func": load_txt_data,
        }

    elif "rcv1x" in experiment:
        test_path = {
            "path": "datasets/rcv1x/rcv1x_test.txt",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/rcv1x/rcv1x_train.txt",
            "load_func": load_txt_data,
        }

    elif "eurlex" in experiment:
        test_path = {
            "path": "datasets/eurlex/eurlex_test.txt",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/eurlex/eurlex_train.txt",
            "load_func": load_txt_data,
        }

    elif "amazoncat" in experiment:
        test_path = {
            "path": "datasets/amazonCat/amazonCat_test.txt",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/amazonCat/amazonCat_train.txt",
            "load_func": load_txt_data,
        }

    elif "wiki10" in experiment:
        test_path = {
            "path": "datasets/wiki10/wiki10_test.txt",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/wiki10/wiki10_train.txt",
            "load_func": load_txt_data,
        }

    elif "wikilshtc" in experiment:
        test_path = {
            "path": "datasets/wikiLSHTC/wikiLSHTC_test.txt",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/wikiLSHTC/wikiLSHTC_train.txt",
            "load_func": load_txt_data,
        }

    elif "amazon" in experiment:
        test_path = {
            "path": "datasets/amazon/amazon_test.txt",
            "load_func": load_txt_data,
        }
        train_path = {
            "path": "datasets/amazon/amazon_train.txt",
            "load_func": load_txt_data,
        }

    else:
        raise ValueError("Unknown experiment")

    if "routers21578" in experiment:
        pass

    # Create binary files for faster loading
    # with Timer():
    #     X_test, Y_test = load_cache_npz_file(**train_path)

    # with Timer():
    #     X_train, Y_train = load_cache_npz_file(**test_path)

    print("Loading data ...")
    print("  Train ...")
    X_train, Y_train = load_libsvm_file(
        train_path["path"], labels_format="csr_matrix", sort_indices=True
    )
    print("  Test ...")
    X_test, Y_test = load_libsvm_file(
        test_path["path"], labels_format="csr_matrix", sort_indices=True
    )

    print(f"Y_train before processing: type={type(Y_train)}, shape={Y_train.shape}")
    print(f"Y_test before processing: type={type(Y_test)}, shape={Y_test.shape}")
    align_dim1(Y_train, Y_test)

    Y_all = sp.vstack((Y_train, Y_test))
    print(
        f"Avg labels per point: {np.mean(Y_train.sum(axis=1))}, avg. samples per label: {np.mean(Y_train.sum(axis=0))}"
    )
    print(
        f"Avg labels per point: {Y_train.sum() / Y_train.shape[0]}, avg. samples per label: {Y_train.sum() / Y_train.shape[1]}"
    )

    # Calculate marginals and propensities
    with Timer():
        print("Calculating marginals and propensities ...")
        marginals = labels_priors(Y_train)
        inv_ps = jpv_inverse_propensity(Y_train)

    print(f"marginals: type={type(marginals)}, shape={marginals.shape}")
    print(f"inv. propensities: type={type(inv_ps)}, shape={inv_ps.shape}")

    print("  Splitting to train and validation ...")
    if testsplit != 0:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=testsplit, random_state=seed
        )
    else:
        X_val, Y_val = X_train, Y_train
    print("  Done")

    print(f"Y_train: type={type(Y_train)}, shape={Y_train.shape}")
    print(f"Y_val: type={type(Y_val)}, shape={Y_val.shape}")
    print(f"Y_test: type={type(Y_test)}, shape={Y_test.shape}")

    print("Training model on a new train data ...")
    model_seed = 42
    model_path = f"models_and_predictions/{experiment}_seed={model_seed}_split={1 - testsplit}_model"
    model = None

    if "splt" in experiment:
        model = StackedNapkinModel(
            model_path,
            model_seed,
            PLT(
                model_path + "_first",
                verbose=True,
                threads=15,
                seed=model_seed,
                max_leaves=200,
                liblinear_eps=0.001,
                liblinear_c=10,
            ),
            PLT(
                model_path + "_second",
                verbose=True,
                threads=15,
                seed=model_seed,
                max_leaves=200,
                liblinear_eps=0.1,
                liblinear_c=10,
            ),
        )
    elif "plt" in experiment:
        model = NapkinModel(
            model_path,
            model_seed,
            PLT(
                model_path,
                verbose=True,
                threads=15,
                seed=model_seed,
                max_leaves=200,
                # liblinear_eps=0.001,
                # liblinear_c=16,
                liblinear_eps=0.01,
                liblinear_c=16,
                # ensemble=3,
            ),
        )
    elif "sbr" in experiment:
        model = StackedNapkinModel(
            model_path,
            model_seed,
            BR(
                model_path + "_first",
                verbose=True,
                threads=15,
                seed=model_seed,
                liblinear_eps=0.001,
                liblinear_c=16,
            ),
            BR(
                model_path + "_second",
                verbose=True,
                threads=15,
                seed=seed,
                liblinear_eps=0.001,
                liblinear_c=16,
            ),
        )
    elif "br" in experiment:
        model = NapkinModel(
            model_path,
            model_seed,
            BR(
                model_path,
                verbose=True,
                threads=15,
                seed=model_seed,
                liblinear_eps=0.001,
                liblinear_c=16,
            ),
        )

    elif "pytorch" in experiment:
        model_seed = 21
        model_path = f"models_and_predictions/{experiment}_seed={model_seed}_split={1 - testsplit}_model"

        if "bce" in experiment:
            model = PytorchModel(model_path, model_seed, loss="bce")
        elif "focal" in experiment:
            model = PytorchModel(model_path, model_seed, loss="focal")
        elif "asym" in experiment:
            model = PytorchModel(model_path, model_seed, loss="asym")

    if isinstance(model, ModelWrapper):
        if not os.path.exists(model_path) or retrain_model:
            with Timer():
                model.fit(X_train, Y_train, X_test, Y_test)
        # else:
        #     model.load()  # Model will load automatically if needed
        print("  Done")

        # top_k = min(1000, Y_train.shape[1])
        top_k = min(200, Y_train.shape[1])
        if "plt" in experiment:
            top_k = 100
        print("Predicting for validation set ...")
        val_pred_path = f"models_and_predictions/{experiment}_seed={model_seed}_split={1 - testsplit}_top_k={top_k}_pred_val.pkl"
        if not os.path.exists(val_pred_path) or retrain_model:
            with Timer():
                pred_val = model.predict_proba(X_val, top_k=top_k)
                align_dim1(Y_train, pred_val)
                # save_npz_wrapper(val_pred_path, pred_val)
                save_pickle(val_pred_path, pred_val)
        else:
            # pred_val = load_npz_wrapper(val_pred_path)
            pred_val = load_pickle(val_pred_path)
        print("  Done")

        print("Predicting for test set ...")
        test_pred_path = f"models_and_predictions/{experiment}_seed={model_seed}_split={1 - testsplit}_top_k={top_k}_pred_test.pkl"
        if not os.path.exists(test_pred_path) or retrain_model:
            with Timer():
                pred_test = model.predict_proba(X_test, top_k=top_k)
                align_dim1(Y_train, pred_test)
                # save_npz_wrapper(test_pred_path, pred_test)
                save_pickle(test_pred_path, pred_test)
        else:
            # pred_test = load_npz_wrapper(test_pred_path)
            pred_test = load_pickle(test_pred_path)
        print("  Done")
        del model

    else:
        pred_val = Y_val.toarray()
        pred_test = Y_test.toarray()
        pred_val = np.clip(pred_val, 1e-8, 1 - 1e-8)
        pred_test = np.clip(pred_test, 1e-8, 1 - 1e-8)
        pred_val = sp.csr_matrix(pred_val)
        pred_test = sp.csr_matrix(pred_test)

    print("Calculating metrics ...")
    output_path_prefix = f"{results_dir}/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for method, func in methods.items():
        print(f"{experiment} - {method} @ {k}: ")

        output_path = f"{output_path_prefix}{method}_k={k}_s={seed}_t={testsplit}"
        results_path = f"{output_path}_results.json"
        pred_path = f"{output_path}_pred.pkl"

        if not os.path.exists(results_path) or recalculate_results:
            results = {}
            if not os.path.exists(pred_path) or recalculate_predictions:
                # results["test_log_loss"] = log_loss(Y_test, pred_test)
                # results["val_log_loss"] = log_loss(Y_val, pred_val)

                y_pred, meta = func[0](
                    Y_val,
                    pred_val,
                    pred_test,
                    k=k,
                    priors=marginals,
                    inv_ps=inv_ps,
                    seed=seed,
                    X_val=X_val,
                    X_test=X_test,
                    model_path=output_path,
                    **func[1],
                )
                results.update(meta)
                print(f"  Iters: {meta['iters']}")
                print(f"  Time: {meta['time']:>5.2f} s")
                # save_npz_wrapper(pred_path, y_pred)
                save_pickle(pred_path, y_pred)
                save_json(results_path, results)
            else:
                # y_pred = load_npz_wrapper(pred_path)
                y_pred = load_pickle(pred_path)
                results = load_json(results_path)

            print("  Calculating metrics:")
            results.update(report_metrics(Y_test, y_pred, k))
            save_json(results_path, results)

        print("  Done")


if __name__ == "__main__":
    main()
