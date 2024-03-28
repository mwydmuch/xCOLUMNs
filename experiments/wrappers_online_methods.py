import numpy as np


from xcolumns.block_coordinate import *
from xcolumns.metrics import *
from xcolumns.weighted_prediction import *
from xcolumns.frank_wolfe import *
from xcolumns.metrics_on_conf_matrix import *
from xcolumns.default_types import *
from xcolumns.utils import *
from tqdm import tqdm


from custom_utilities_methods import *
from wrappers_threshold_methods import *


def online_greedy_wrapper(y_proba, bin_utility_func, k: int = 5, seed: int = 0, y_true=None, repeats=1, **kwargs):
    y_preds = []
    np.random.seed(seed)
    
    # Get specialized functions
    if isinstance(y_proba, np.ndarray):
        bc_with_0approx_step_func = bc_with_0approx_np_step
        random_at_k_func = random_at_k_np
    elif isinstance(y_proba, csr_matrix):
        bc_with_0approx_step_func = bc_with_0approx_csr_step
        random_at_k_func = random_at_k_csr
    else:
        raise ValueError("y_proba must be either np.ndarray or csr_matrix")
    
    n, m = y_proba.shape

    rng = np.random.default_rng(seed)
    order = np.arange(n)
    
    for r in range(1, repeats + 1):
        print(f"  Starting {r} try of online greedy")
        rng.shuffle(order)
        meta = {"time": time(), "iters": 1}
        y_pred = random_at_k_func((n, m), k=k, seed=seed)

        tp = np.zeros(m, dtype=FLOAT_TYPE)
        fp = np.zeros(m, dtype=FLOAT_TYPE)
        fn = np.zeros(m, dtype=FLOAT_TYPE)
        tn = np.zeros(m, dtype=FLOAT_TYPE)

        for i in tqdm(order):
            bc_with_0approx_step_func(
                y_proba,
                y_pred,
                i,
                tp,
                fp,
                fn,
                tn,
                k,
                bin_utility_func,
                greedy=True,
                skip_tn=False,
                only_pred=True,
            )
            
            tp += y_pred[i] * y_true[i]
            fp += y_pred[i] * (1 - y_true[i])
            fn += (1 - y_pred[i]) * y_true[i]
            tn += (1 - y_pred[i]) * (1 - y_true[i])

        meta["time"] = time() - meta["time"]
        y_preds.append(y_pred)

    return y_preds, meta


def online_greedy_macro_f1(y_proba, k: int = 5, seed: int = 0, y_true=None, repeats=1, **kwargs):
    return online_greedy_wrapper(y_proba, bin_fmeasure_on_conf_matrix, k=k, seed=seed, y_true=y_true, repeats=repeats, **kwargs)


def online_greedy_macro_min_tp_tn(y_proba, k: int = 5, seed: int = 0, y_true=None, repeats=1, **kwargs):
    return online_greedy_wrapper(y_proba, bin_min_tp_tn, k=k, seed=seed, y_true=y_true, repeats=repeats, **kwargs)


def online_frank_wolfe_wrapper(
    y_proba, utility_func, k: int = 5, seed: int = 0, y_true=None, repeats=1, epochs=1, 
    update: int = 100,
    exp_update: bool = True,
    **kwargs
):
    y_preds = []
    np.random.seed(seed)
    
    n, m = y_proba.shape

    rng = np.random.default_rng(seed)
    order = np.arange(n)
    
    for r in range(1, repeats + 1):
        print(f"  Starting {r} try of fw online")
        rng.shuffle(order)
        meta = {"time": time(), "iters": 1}

        classifiers_a = np.zeros((1, m), dtype=FLOAT_TYPE)
        classifiers_b = np.zeros((1, m), dtype=FLOAT_TYPE)
        classifiers_proba = np.ones(1, dtype=FLOAT_TYPE)

        classifiers_a[0] = np.ones(m, dtype=FLOAT_TYPE)
        classifiers_b[0] = np.full(m, -0.5, dtype=FLOAT_TYPE)

        for e in range(epochs):
            print(f"  epoch {e + 1} / {epochs}")
            y_pred = np.zeros((n, m), dtype=FLOAT_TYPE)

            u = update
            i = 0
            while i < n:
                print(f"  batch {i} - {i + u} / {n}")
                batch = order[i:min(i + u, n)]
                all_till_now = order[:i + u]
                y_pred[batch] = predict_using_randomized_classifier(
                    y_proba[batch], classifiers_a, classifiers_b, classifiers_proba, k=k, seed=seed + i
                )

                if i + u < n:
                    classifiers_a, classifiers_b, classifiers_proba, _ = find_optimal_randomized_classifier_using_frank_wolfe(
                        y_true[all_till_now], y_proba[all_till_now], utility_func, max_iters=20, k=k, **kwargs
                    )
                i += u
                if exp_update:
                    u *= 2

        meta["time"] = time() - meta["time"]
        y_preds.append(y_pred)

    return y_preds, meta


def online_fw_macro_f1(y_proba, k: int = 5, seed: int = 0, y_true=None, repeats=1, **kwargs):
    return online_frank_wolfe_wrapper(y_proba, macro_fmeasure_on_conf_matrix, k=k, seed=seed, y_true=y_true, repeats=repeats, **kwargs)


def online_find_thresholds_wrapper(
    y_proba, utility_func, k: int = 5, seed: int = 0, y_true=None, repeats=1, 
    update: int = 100,
    exp_update: bool = True,
    **kwargs
):
    y_preds = []
    np.random.seed(seed)
    
    n, m = y_proba.shape

    rng = np.random.default_rng(seed)
    order = np.arange(n)
    
    for r in range(1, repeats + 1):
        u = update

        print(f"  Starting {r} try of thresholds online")
        rng.shuffle(order)
        meta = {"time": time(), "iters": 1}

        thresholds = np.full(m, 0.5, dtype=FLOAT_TYPE)
        y_pred = np.zeros((n, m), dtype=FLOAT_TYPE)

        i = 0
        while i < n:
            print(f"  batch {i} - {i + u} / {n}")
            batch = order[i:min(i + u, n)]
            all_till_now = order[:i + u]

            if k > 0:
                y_pred[batch] = np.zeros(y_pred[batch].shape, dtype=FLOAT_TYPE)
                for j in range(u):
                    gains = y_pred[batch[i + j]] - thresholds
                    top_k = np.argpartition(-gains, k)[:k]
                    y_pred[batch[i + j], top_k] = 1.0
            else:
                y_pred[batch] = y_proba[batch] >= thresholds

            if i + u < n:
                thresholds, _ = find_thresholds(
                    y_true[all_till_now], y_proba[all_till_now], utility_func
                )
            i += u
            if exp_update:
                u *= 2

        meta["time"] = time() - meta["time"]
        y_preds.append(y_pred)

    return y_preds, meta


def online_thresholds_macro_f1(y_proba, k: int = 5, seed: int = 0, y_true=None, repeats=1, **kwargs):
    return online_find_thresholds_wrapper(y_proba, bin_fmeasure_on_conf_matrix, k=k, seed=seed, y_true=y_true, repeats=repeats, **kwargs)


def online_thresholds_macro_min_tp_tn(y_proba, k: int = 5, seed: int = 0, y_true=None, repeats=1, **kwargs):
    return online_find_thresholds_wrapper(y_proba, bin_min_tp_tn, k=k, seed=seed, y_true=y_true, repeats=repeats, **kwargs)


def online_gd_on_conf_mat_wrapper(
    y_proba, utility_func, k: int = 5, seed: int = 0, y_true=None, repeats=1, epochs=1, 
    **kwargs
):
    y_preds = []
    np.random.seed(seed)
    
    n, m = y_proba.shape

    rng = np.random.default_rng(seed)
    order = np.arange(n)
    
    
    for r in range(1, repeats + 1):

        print(f"  Starting {r} try of online gd on C")
        meta = {"time": time(), "iters": 1}
        
        rng.shuffle(order)
        
        classifier_a = np.ones(m, dtype=FLOAT_TYPE)
        classifier_b = np.full(m, -0.5, dtype=FLOAT_TYPE)
        
        tp = np.zeros(m, dtype=FLOAT_TYPE)
        fp = np.zeros(m, dtype=FLOAT_TYPE)
        fn = np.zeros(m, dtype=FLOAT_TYPE)
        tn = np.zeros(m, dtype=FLOAT_TYPE)

        for e in range(epochs):
            print(f"  epoch {e + 1} / {epochs}")
            print_update = 100
            y_pred = np.zeros((n, m), dtype=FLOAT_TYPE)

            for j, i in enumerate(order):

                gains = y_proba[i] * classifier_a + classifier_b
                if k > 0:
                    top_k = np.argpartition(-gains, k)[:k]
                    y_pred[i, top_k] = 1.0
                else:
                    y_pred[i, gains > 0.0] = 1.0

                tp += y_pred[i] * y_true[i]
                fp += y_pred[i] * (1 - y_true[i])
                fn += (1 - y_pred[i]) * y_true[i]
                tn += (1 - y_pred[i]) * (1 - y_true[i])

                utility, Gtp, Gfp, Gfn, Gtn = utility_func_with_gradient(utility_func, tp, fp, fn, tn)
                classifier_a = Gtp - Gfp - Gfn + Gtn
                classifier_b = Gfp - Gtn

                if j % print_update == print_update - 1:
                    print_update *= 2
                    print(f"    {j + 1} / {n}, utility so far: {utility}")

        meta["time"] = time() - meta["time"]
        y_preds.append(y_pred)

    return y_preds, meta


def online_gd_macro_f1(y_proba, k: int = 5, seed: int = 0, y_true=None, repeats=1, **kwargs):
    return online_gd_on_conf_mat_wrapper(y_proba, macro_fmeasure_on_conf_matrix, k=k, seed=seed, y_true=y_true, repeats=repeats, **kwargs)