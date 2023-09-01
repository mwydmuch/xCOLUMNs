import numpy as np
import scipy.sparse as sp

from metrics import *
from data import *
from utils_misc import *
from frank_wolfe import *
from weighted_prediction import *
from sklearn.model_selection import train_test_split
from napkinxc.models import PLT
from napkinxc.datasets import to_csr_matrix, load_libsvm_file

import sys
import click

RECALCULATE_RESUTLS = False
RECALCULATE_PREDICTION = False
RETRAIN_MODEL = False
K = (1, 3, 5, 10)


def frank_wolfe_wrapper(Y_val, pred_val, pred_test, utility_func, k: int = 5, seed: int = 0, reg=0, pred_repeat=10, average=False, **kwargs):
    classifiers, classifier_weights = frank_wolfe(Y_val, pred_val, utility_func, max_iters=20, k=k, reg=reg, **kwargs)
    print(f"  classifiers weights: {classifier_weights}")
    y_preds = []
    if not average:
        print("  predicting with randomized classfier")
        for i in range(pred_repeat):
            y_pred = predict_top_k_for_classfiers(pred_test, classifiers, classifier_weights, k=k, seed=seed + i)
            y_preds.append(y_pred)
    else:
        print("  averaging classifiers weights")
        avg_classifier_weights = np.zeros((classifiers.shape[1], classifiers.shape[2]))
        for i in range(classifier_weights.shape[0]):
            avg_classifier_weights += classifier_weights[i] * classifiers[i]
        avg_classifier_weights /= classifier_weights.shape[0]
        y_pred = predict_top_k(pred_test, avg_classifier_weights, k)
        y_preds.append(y_pred)

    return y_preds


def frank_wolfe_macro_recall(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return frank_wolfe_wrapper(Y_val, pred_val, pred_test, macro_recall_C, k=k, seed=seed, **kwargs)


def frank_wolfe_macro_precision(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return frank_wolfe_wrapper(Y_val, pred_val, pred_test, macro_precision_C, k=k, seed=seed, **kwargs)


def frank_wolfe_macro_f1(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return frank_wolfe_wrapper(Y_val, pred_val, pred_test, macro_f1_C, k=k, seed=seed, **kwargs)


def report_metrics(data, predictions, k):
    results = {}
    for metric, func in METRICS.items():
        values = []
        for pred in predictions:
            value = func(data, pred)
            values.append(value)
        results[f"{metric}@{k}"] = values
        print(f"  {metric}: {100 * np.mean(values):>5.2f} +/- {100 * np.std(values):>5.2f}")

    return results


def fw_optimal_instance_precision_wrapper(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return [optimal_instance_precision(pred_test, k=k, **kwargs)]

def fw_optimal_macro_recal_wrapper(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return [optimal_macro_recall(pred_test, k=k, **kwargs)]

def fw_power_law_weighted_instance_wrapper(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return [power_law_weighted_instance(pred_test, k=k, **kwargs)]

def fw_log_weighted_instance_wrapper(Y_val, pred_val, pred_test, k: int = 5, seed: int = 0, **kwargs):
    return [log_weighted_instance(pred_test, k=k, **kwargs)]


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
    "fw-split-power-law-with-beta=0.5": (fw_power_law_weighted_instance_wrapper,{"beta": 0.5}),
    "fw-split-power-law-with-beta=0.25": (fw_power_law_weighted_instance_wrapper,{"beta": 0.25}),
    "fw-split-log": (fw_log_weighted_instance_wrapper,{}),
    "frank-wolfe-macro-recall": (frank_wolfe_macro_recall, {}),
    "frank-wolfe-macro-precision": (frank_wolfe_macro_precision, {}),
    "frank-wolfe-macro-f1": (frank_wolfe_macro_f1, {}),
    # "frank-wolfe-macro-recall-avg": (frank_wolfe_macro_recall, {"average": True}),
    # "frank-wolfe-macro-precision-avg": (frank_wolfe_macro_precision, {"average": True}),
    # "frank-wolfe-macro-f1-avg": (frank_wolfe_macro_f1, {"average": True}),
    # "frank-wolfe-macro-recall-rnd": (frank_wolfe_macro_recall, {"init": "random"}),
    # "frank-wolfe-macro-precision-rnd": (frank_wolfe_macro_precision, {"init": "random"}),
    # "frank-wolfe-macro-f1-rnd": (frank_wolfe_macro_f1, {"init": "random"}),
}


def fix_shape(csr_a, csr_b):
    if csr_a.shape[1] != csr_b.shape[1]:
        print("  Fixing shapes ...")
        new_size = max(csr_a.shape[1], csr_b.shape[1])
        csr_a.resize((csr_a.shape[0], new_size))
        csr_b.resize((csr_b.shape[0], new_size))


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



class PLTModel(ModelWrapper):
    def __init__(self, model_path, seed):
        super().__init__(model_path, seed)
        self.model = PLT(model_path, verbose=True, threads=15, seed=seed, max_leaves=200, liblinear_eps=0.001, liblinear_c=16)
    
    def fit(self, X, Y):
        self.model.fit(X, Y)
    
    def predict_proba(self, X, top_k):
        pred_val = self.model.predict_proba(X, top_k=top_k)
        pred_val = to_csr_matrix(pred_val, sort_indices=True)
        return pred_val


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("-k", type=int, required=False, default=None)
@click.option("-s", "--seed", type=int, required=False, default=None)
@click.option("-t", "--testsplit", type=float, required=False, default=0)
@click.option("-r", "--reg", type=float, required=False, default=0)
def main(experiment, k, seed, testsplit, reg):
    print(experiment)

    if k is not None:
        K = (k,)
    
    lightxml_data_load_config = {"labels_delimiter": " ", "labels_features_delimiter": None, "header": False}
    xmlc_data_load_config = {"labels_delimiter": ",", "labels_features_delimiter": " ", "header": True}

    if "yeast" in experiment:
        xmlc_data_load_config["header"] = False
        test_path = {"path": "datasets/yeast/yeast_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/yeast/yeast_train.txt", "load_func": load_txt_data}

    elif "youtube_deepwalk" in experiment:
        xmlc_data_load_config["header"] = False
        test_path = {"path": "datasets/youtube_deepwalk/youtube_deepwalk_test.svm", "load_func": load_txt_data}
        train_path = {"path": "datasets/youtube_deepwalk/youtube_deepwalk_train.svm", "load_func": load_txt_data}

    elif "eurlex_lexglue" in experiment:
        xmlc_data_load_config["header"] = False
        test_path = {"path": "datasets/eurlex_lexglue/eurlex_lexglue_test.svm", "load_func": load_txt_data}
        train_path = {"path": "datasets/eurlex_lexglue/eurlex_lexglue_train.svm", "load_func": load_txt_data}

    elif "mediamill" in experiment:
        xmlc_data_load_config["header"] = False
        test_path = {"path": "datasets/mediamill/mediamill_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/mediamill/mediamill_train.txt", "load_func": load_txt_data}

    elif "bibtex" in experiment:
        xmlc_data_load_config["header"] = False
        test_path = {"path": "datasets/bibtex/bibtex_test.svm", "load_func": load_txt_data}
        train_path = {"path": "datasets/bibtex/bibtex_train.svm", "load_func": load_txt_data}

    elif "flicker_deepwalk" in experiment:
        xmlc_data_load_config["header"] = False
        test_path = {"path": "datasets/flicker_deepwalk/flicker_deepwalk_test.svm", "load_func": load_txt_data}
        train_path = {"path": "datasets/flicker_deepwalk/flicker_deepwalk_train.svm", "load_func": load_txt_data}

    elif "rcv1x" in experiment:
        test_path = {"path": "datasets/rcv1x/rcv1x_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/rcv1x/rcv1x_train.txt", "load_func": load_txt_data}

    elif "eurlex" in experiment:
        test_path = {"path": "datasets/eurlex/eurlex_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/eurlex/eurlex_train.txt", "load_func": load_txt_data}

    elif "amazoncat" in experiment:
        test_path = {"path": "datasets/amazonCat/amazonCat_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/amazonCat/amazonCat_train.txt", "load_func": load_txt_data}

    elif "wiki10" in experiment:
        test_path = {"path": "datasets/wiki10/wiki10_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/wiki10/wiki10_train.txt", "load_func": load_txt_data}

    elif "amazon" in experiment:
        test_path = {"path": "datasets/amazon/amazon_test.txt", "load_func": load_txt_data}
        train_path = {"path": "datasets/amazon/amazon_train.txt", "load_func": load_txt_data}


    if "plt" in experiment:
        model = PLTModel()

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

    Y_all = sp.vstack((Y_train, Y_test))
    print(f"Avg labels per point: {np.mean(Y_train.sum(axis=1))}, avg. samples per label: {np.mean(Y_train.sum(axis=0))}")
    print(f"Avg labels per point: {Y_train.sum() / Y_train.shape[0]}, avg. samples per label: {Y_train.sum() / Y_train.shape[1]}")
    
    # For some sparse format this resize might be necessary
    fix_shape(Y_train, Y_test)

    # Calculate marginals and propensities
    with Timer():
        print("Calculating marginals and propensities ...")
        marginals = labels_priors(Y_train)
        inv_ps = jpv_inverse_propensity(Y_train)

    print("  Spliting to train and validation ...")
    if testsplit != 0:
        X_train,  X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=testsplit, random_state=seed)
    else:
        X_val, Y_val = X_train, Y_train
    print("  Done")

    print(f"Y_train: type={type(Y_train)}, shape={Y_train.shape}")
    print(f"Y_val: type={type(Y_val)}, shape={Y_val.shape}")
    print(f"Y_test: type={type(Y_test)}, shape={Y_test.shape}")

    print("Training model on splited train data ...")
    model_path = f"models_and_predictions/{experiment}_seed={seed}_split={1 - testsplit}_model"
    
    if not os.path.exists(os.path.join(model_path, "weights.bin")) or RETRAIN_MODEL:
        with Timer():
            model.fit(X_train, Y_train)
    else:
        model.load()  # Model will load automatically if needed
    print("  Done")

    #top_k = min(1000, Y_train.shape[1])
    top_k = min(200, Y_train.shape[1])
    print("Predicting for validation set ...")
    val_pred_path = f"models_and_predictions/{experiment}_seed={seed}_split={1 - testsplit}_top_k={top_k}_pred_val.pkl"
    if not os.path.exists(val_pred_path) or RETRAIN_MODEL:
        with Timer():
            pred_val = model.predict_proba(X_val, top_k=top_k)
            fix_shape(Y_train, pred_val)
            #save_npz_wrapper(val_pred_path, pred_val)
            save_pickle(val_pred_path, pred_val)
    else:
        #pred_val = load_npz_wrapper(val_pred_path)
        pred_val = load_pickle(val_pred_path)
    print("  Done")

    print("Predicting for test set ...")
    test_pred_path = f"models_and_predictions/{experiment}_seed={seed}_split={1 - testsplit}_top_k={top_k}_pred_test.pkl"
    if not os.path.exists(test_pred_path) or RETRAIN_MODEL:
        with Timer():
            pred_test = model.predict_proba(X_test, top_k=top_k)
            fix_shape(Y_train, pred_test)
            #save_npz_wrapper(test_pred_path, pred_test)
            save_pickle(test_pred_path, pred_test)
    else:
        #pred_test = load_npz_wrapper(test_pred_path)
        pred_test = load_pickle(test_pred_path)
    print("  Done")

    print("Calculating metrics ...")
    output_path_prefix = f"results/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for k in K:
        for method, func in METHODS.items():
            print(f"{method} @ {k}: ")

            output_path = f"{output_path_prefix}{method}_k={k}_s={seed}_t={testsplit}_r={reg}"
            results_path = f"{output_path}_results.json"
            pred_path = f"{output_path}_pred.pkl"
            
            if not os.path.exists(results_path) or RECALCULATE_RESUTLS:
                results = {}
                if not os.path.exists(pred_path) or RECALCULATE_PREDICTION:
                    with Timer() as t:
                        y_preds = func[0](Y_val, pred_val, pred_test, k=k, marginals=marginals, inv_ps=inv_ps, seed=seed, reg=reg, **func[1])
                        results["time"] = t.get_time()
                    #save_npz_wrapper(pred_path, y_pred)
                    save_pickle(pred_path, y_preds)
                    save_json(results_path, results)
                else:
                    #y_pred = load_npz_wrapper(pred_path)
                    y_preds = load_pickle(pred_path)
                    results = load_json(results_path)
                
                print("  Calculating metrics:")
                results.update(report_metrics(Y_test, y_preds, k))
                save_json(results_path, results)

            print("  Done")


if __name__ == "__main__":
    main()
