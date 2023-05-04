import numpy as np

from metrics import *
from data import *
from utils import *
from frank_wolfe import *
from prediction import *
from sklearn.model_selection import train_test_split
from napkinxc.models import PLT
from napkinxc.datasets import to_csr_matrix, load_libsvm_file

import sys
import click

RECALCULATE_RESUTLS = True
RECALCULATE_PREDICTION = True
RETRAIN_MODEL = False
K = (1, 3, 5, 10)


def frank_wolfe_wrapper(Y_val, pred_val, pred_test, loss_func, k: int = 5, seed: int = 0, **kwargs):
    classifiers, classifier_weights = frank_wolfe(Y_val, pred_val.toarray(), max_iters=50, loss_func=loss_func, k=k)
    print("  Classifier_weights: ", classifier_weights)
    y_pred = predict_top_k_for_classfiers(pred_test.toarray(), classifiers, classifier_weights, k=k, seed=seed)
    return y_pred


def frank_wolfe_macro_recall(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return frank_wolfe_wrapper(Y_val, pred_val, pred_test, fw_macro_recall, k=k, seed=seed)


def frank_wolfe_macro_precision(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return frank_wolfe_wrapper(Y_val, pred_val, pred_test, fw_macro_precision, k=k, seed=seed)


def frank_wolfe_macro_f1(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return frank_wolfe_wrapper(Y_val, pred_val, pred_test, fw_macro_f1, k=k, seed=seed)


def report_metrics(data, predictions, k):
    results = {}
    for metric, func in METRICS.items():
        value = func(data, predictions)
        results[f"{metric}@{k}"] = value
        print(f"  {metric}: {100 * func(data, predictions):>5.2f}")

    return results


def fw_optimal_instance_precision_wrapper(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return optimal_instance_precision(pred_test, k=k, **kwargs)

def fw_optimal_macro_recal_wrapper(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return optimal_macro_recall(pred_test, k=k, **kwargs)


METRICS = {
    "mC": macro_abandonment,
    "iC": instance_abandonment, 
    "mP": macro_precision,
    "iP": instance_precision,
    "mR": macro_recall,
    "iR": instance_recall,
    "mF": macro_f1,
    "iF": instance_f1
}


METHODS = {
    "fw-split-optimal-instance-prec": (fw_optimal_instance_precision_wrapper, {}),
    "fw-split-optimal-macro-recall": (fw_optimal_macro_recal_wrapper, {}),
    "frank-wolfe-macro-recall": (frank_wolfe_macro_recall, {}),
    "frank-wolfe-macro-precision": (frank_wolfe_macro_precision, {}),
    "frank-wolfe-macro-f1": (frank_wolfe_macro_f1, {}),
}


def fix_shape(csr_a, csr_b):
    if csr_a.shape[1] != csr_b.shape[1]:
        print("  Fixing shapes ...")
        new_size = max(csr_a.shape[1], csr_b.shape[1])
        csr_a.resize((csr_a.shape[0], new_size))
        csr_b.resize((csr_b.shape[0], new_size))


def load_txt_data():
    pass


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("-k", type=int, required=False, default=None)
@click.option("-s", "--seed", type=int, required=False, default=None)
def main(experiment, k, seed):    
    print(experiment)

    if k is not None:
        K = (k,)
    
    lightxml_data_load_config = {"labels_delimiter": " ", "labels_features_delimiter": None, "header": False}
    xmlc_data_load_config = {"labels_delimiter": ",", "labels_features_delimiter": " ", "header": True}

    if "yeast_plt" in experiment:
        # yeast - PLT
        xmlc_data_load_config["header"] = False
        test_path = {"path": "datasets/yeast/yeast_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/yeast/yeast_train.txt", "load_func": load_txt_data}

    elif "mediamill_plt" in experiment:
        # mediamill - PLT
        xmlc_data_load_config["header"] = False
        test_path = {"path": "datasets/mediamill/mediamill_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/mediamill/mediamill_train.txt", "load_func": load_txt_data}

    elif "rcv1x_plt" in experiment:
        # RCV1X - PLT + XMLC repo data
        test_path = {"path": "datasets/rcv1x/rcv1x_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/rcv1x/rcv1x_train.txt", "load_func": load_txt_data}

    elif "eurlex_plt" in experiment:
        test_path = {"path": "datasets/eurlex/eurlex_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/eurlex/eurlex_train.txt", "load_func": load_txt_data}

    elif "amazoncat_plt" in experiment:
        test_path = {"path": "datasets/amazonCat/amazonCat_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/amazonCat/amazonCat_train.txt", "load_func": load_txt_data}

    elif "wiki10_plt" in experiment:
        test_path = {"path": "datasets/wiki10/wiki10_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/wiki10/wiki10_train.txt", "load_func": load_txt_data}

    elif "amazon_plt" in experiment:
        test_path = {"path": "datasets/amazon/amazon_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/amazon/amazon_train.txt", "load_func": load_txt_data}

    # Create binary files for faster loading
    # with Timer():
    #     X_test, Y_test = load_cache_npz_file(**train_path)

    # with Timer():
    #     X_train, Y_train = load_cache_npz_file(**test_path)

    print("Loading data ...")
    print("  Train ...")
    X_train, Y_train = load_libsvm_file(train_path["path"], labels_format="csr_matrix", sort_indices=True)
    print("  Test ...")
    X_test, Y_test = load_libsvm_file(test_path["path"], labels_format="csr_matrix", sort_indices=True)

    print(f"Y_train before processing: type={type(Y_train)}, shape={Y_train.shape}")
    print(f"Y_test before processing: type={type(Y_test)}, shape={Y_test.shape}")
    
    # For some sparse format this resize might be necessary
    fix_shape(Y_train, Y_test)

    # Calculate marginals and propensities
    with Timer():
        print("Calculating marginals and propensities ...")
        marginals = labels_priors(Y_train)
        inv_ps = jpv_inverse_propensity(Y_train)

    print("  Spliting to train and validation ...")
    X_train,  X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.5, random_state=seed)
    print("  Done")

    print(f"Y_train: type={type(Y_train)}, shape={Y_train.shape}")
    print(f"Y_val: type={type(Y_val)}, shape={Y_val.shape}")
    print(f"Y_test: type={type(Y_test)}, shape={Y_test.shape}")

    print("Training model on splited train data ...")
    model_path = f"models_and_predictions/{experiment}_seed={seed}_model"
    
    
    model = PLT(model_path, verbose=True, threads=12, seed=seed)
    if not os.path.exists(os.path.join(model_path, "weights.bin")) or RETRAIN_MODEL:
        with Timer():
            model.fit(X_train, Y_train)
    else:
        model.load()
    print("  Done")

    top_k = min(1000, Y_train.shape[1])
    top_k = min(200, Y_train.shape[1])
    print("Predicting for validation set ...")
    val_pred_path = f"models_and_predictions/{experiment}_seed={seed}_top_k={top_k}_pred_val.npz"
    if not os.path.exists(val_pred_path) or RETRAIN_MODEL:
        with Timer():
            pred_val = model.predict_proba(X_val, top_k=top_k)
            pred_val = to_csr_matrix(pred_val)
            fix_shape(Y_train, pred_val)
            save_npz_wrapper(val_pred_path, pred_val)
    else:
        pred_val = load_npz_wrapper(val_pred_path)
    print("  Done")

    print("Predicting for test set ...")
    test_pred_path = f"models_and_predictions/{experiment}_seed={seed}_top_k={top_k}_pred_test.npz"
    if not os.path.exists(test_pred_path) or RETRAIN_MODEL:
        with Timer():
            pred_test = model.predict_proba(X_test, top_k=top_k)
            pred_test = to_csr_matrix(pred_test)
            fix_shape(Y_train, pred_test)
            save_npz_wrapper(test_pred_path, pred_test)
    else:
        pred_test = load_npz_wrapper(test_pred_path)
    print("  Done")

    # print(Y_test[0], pred_test[0])
    # exit(1)

    print("Calculating metrics ...")
    output_path_prefix = f"results/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for k in K:
        for method, func in METHODS.items():
            print(f"{method} @ {k}: ")

            output_path = f"{output_path_prefix}{method}_k={k}_s={seed}"
            results_path = f"{output_path}_results.json"
            pred_path = f"{output_path}_pred.npz"
            
            if not os.path.exists(results_path) or RECALCULATE_RESUTLS:
                results = {}
                if not os.path.exists(pred_path) or RECALCULATE_PREDICTION:
                    with Timer() as t:
                        y_pred = func[0](Y_val, pred_val, pred_test, k=k, marginals=marginals, inv_ps=inv_ps, seed=seed, **func[1])
                        results["time"] = t.get_time()
                    save_npz_wrapper(pred_path, y_pred)
                    save_json(results_path, results)
                else:
                    y_pred = load_npz_wrapper(pred_path)
                    results = load_json(results_path)
                
                print("  Calculating metrics:")
                results.update(report_metrics(Y_test, y_pred, k))
                save_json(results_path, results)

            print("  Done")


if __name__ == "__main__":
    main()
