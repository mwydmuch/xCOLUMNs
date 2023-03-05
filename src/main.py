import numpy as np

from metrics import *
from data import *
from prediction import *
from utils import *

import sys
import click

ONLY_RESULTS = False
K = (1, 3, 5, 10)

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
    # "random": predict_random_at_k,
    "optimal-instance-prec": (optimal_instance_precision,{}),
    "optimal-instance-ps-prec": (inv_propensity_weighted_instance,{}),
    "power-law-with-beta=0.5": (power_law_weighted_instance,{"beta": 0.5}),
    "power-law-with-beta=0.25": (power_law_weighted_instance,{"beta": 0.25}),
    "optimal-macro-recall-test-marginals": (optimal_macro_recall,{}),
    "optimal-macro-recall": (optimal_macro_recall,{}),
    "log": (log_weighted_instance,{}),
    "block-coord-macro-prec": (block_coordinate_macro_precision,{}),
    "block-coord-macro-f1": (block_coordinate_macro_f1,{}),
    
    "greedy-start-block-coord-macro-prec": (block_coordinate_macro_precision, {"greedy_start": True}),
    "greedy-start-block-coord-macro-f1": (block_coordinate_macro_f1, {"greedy_start": True}),
    "greedy-macro-prec": (block_coordinate_macro_precision, {"greedy_start": True, "max_iter": 1}),
    "greedy-macro-f1": (block_coordinate_macro_f1, {"greedy_start": True, "max_iter": 1}),
    # "block coord coverage": block_coordinate_coverage,

    # Tolerance experiments
    "block-coord-macro-prec-tol=1e-4": (block_coordinate_macro_precision, {"tolerance": 1e-4}),
    "block-coord-macro-f1-tol=1e-4": (block_coordinate_macro_f1,{"tolerance": 1e-4}),
    "block-coord-macro-prec-tol=1e-5": (block_coordinate_macro_precision,{"tolerance": 1e-5}),
    "block-coord-macro-f1-tol=1e-5": (block_coordinate_macro_f1,{"tolerance": 1e-5}),
    "block-coord-macro-prec-tol=1e-6": (block_coordinate_macro_precision,{"tolerance": 1e-6}),
    "block-coord-macro-f1-tol=1e-6": (block_coordinate_macro_f1,{"tolerance": 1e-6}),
}

def report_metrics(data, predictions, k):
    results = {}
    for metric, func in METRICS.items():
        value = func(data, predictions)
        results[f"{metric}@{k}"] = value
        print(f"  {metric}: {100 * func(data, predictions):>5.2f}")

    return results


@click.command()
@click.argument("experiment", type=str, required=True)
@click.option("-k", type=int, required=False, default=None)
@click.option("-s", "--seed", type=int, required=False, default=None)
def main(experiment, k, seed):    

    experiment = "eurlex_plt"
    if len(sys.argv) > 1:
        experiment = sys.argv[1]

    print(experiment)

    if k is not None:
        K = (k,)

    true_as_pred = "true_as_pred" in experiment
    lightxml_data = "lightxml" in experiment

    plt_loss = "log"
    
    lightxml_data_load_config = {"labels_delimiter": " ", "labels_features_delimiter": None, "header": False}
    xmlc_data_load_config = {"labels_delimiter": ",", "labels_features_delimiter": " ", "header": True}

    if "eurlex_plt" in experiment:
        # Eurlex - PLT + XMLC repo data
        eta_pred_path = {"path": f"predictions/eurlex_top_100_{plt_loss}", "load_func": load_txt_sparse_pred}
        y_true_path = {"path": "datasets/eurlex/eurlex_test.txt", "load_func": load_txt_labels}
        train_y_true_path = {"path": "datasets/eurlex/eurlex_train.txt", "load_func": load_txt_labels}

    elif "amazoncat_plt" in experiment:
        # AmazonCat - PLT + XMLC repo data
        eta_pred_path = {"path": f"predictions/amazonCat_top_100_{plt_loss}", "load_func": load_txt_sparse_pred}
        y_true_path = {"path": "datasets/amazonCat/amazonCat_test.txt", "load_func": load_txt_labels}
        train_y_true_path = {"path": "datasets/amazonCat/amazonCat_train.txt", "load_func": load_txt_labels}

    elif "wiki10_plt" in experiment:
        # Wiki10 - PLT + XMLC repo data
        eta_pred_path = {"path": f"predictions/wiki_top_100_{plt_loss}", "load_func": load_txt_sparse_pred}
        y_true_path = {"path": "datasets/wiki10/wiki10_test.txt", "load_func": load_txt_labels}
        train_y_true_path = {"path": "datasets/wiki10/wiki10_train.txt", "load_func": load_txt_labels}

    elif "amazon_plt" in experiment:
        # Amazon - PLT + XMLC repo data
        eta_pred_path = {"path": f"predictions/amazon_top_100_{plt_loss}", "load_func": load_txt_sparse_pred}
        y_true_path = {"path": "datasets/amazon/amazon_test.txt", "load_func": load_txt_labels}
        train_y_true_path = {"path": "datasets/amazon/amazon_train.txt", "load_func": load_txt_labels}
    
    elif "amazon_plt_1000" in experiment:
        # Amazon - PLT + XMLC repo data
        eta_pred_path = {"path": f"predictions/amazon_top_1000_{plt_loss}", "load_func": load_txt_sparse_pred}
        y_true_path = {"path": "datasets/amazon/amazon_test.txt", "load_func": load_txt_labels}
        train_y_true_path = {"path": "datasets/amazon/amazon_train.txt", "load_func": load_txt_labels}



    elif "eurlex_lightxml" in experiment:
        # Eurlex - LightXML
        y_true_path = {"path": "datasets/EUR-Lex/test_labels.txt", "load_func": load_txt_labels}
        eta_pred_path = {"path": "predictions/eurlex/eurlex4k_full_plain-scores.npy", "load_func": load_npy_full_pred, "keep_top_k": 100, "apply_sigmoid": True}
        train_y_true_path = {"path": "datasets/EUR-Lex/train_labels.txt", "load_func": load_txt_labels}

    elif "amazoncat_lightxml" in experiment:
        # Wiki - LightXML
        y_true_path = {"path": "datasets/AmazonCat-13K/test_labels.txt", "load_func": load_txt_labels}
        eta_pred_path = {"path": "predictions/amazonCat_top_100.notnpz", "load_func": load_npz_wrapper, "apply_sigmoid": True}
        train_y_true_path = {"path": "datasets/AmazonCat-13K/train_labels.txt", "load_func": load_txt_labels}

    elif "wiki10_lightxml" in experiment:
        # Wiki - LightXML
        y_true_path = {"path": "datasets/Wiki10-31K/test_labels.txt", "load_func": load_txt_labels}
        eta_pred_path = {"path": "predictions/wiki10_top_100.notnpz", "load_func": load_npz_wrapper, "apply_sigmoid": True}
        train_y_true_path = {"path": "datasets/Wiki10-31K/train_labels.txt", "load_func": load_txt_labels}

    elif "amazon_lightxml" in experiment:
        # Amazon - LightXML
        y_true_path = {"path": "datasets/Amazon-670K/test_labels.txt", "load_func": load_txt_labels}
        eta_pred_path = {"path": "predictions/amazon/amazon670k_light_t0", "load_func": load_npy_sparse_pred}
        train_y_true_path = {"path": "datasets/Amazon-670K/train_labels.txt", "load_func": load_txt_labels}

    elif "amazon_1000_lightxml" in experiment:
        # Amazon - LightXML
        y_true_path = {"path": "datasets/Amazon-670K/test_labels.txt", "load_func": load_txt_labels}
        eta_pred_path = {"path": "predictions/amazon_1000/amazon670k_light_original_t0", "load_func": load_npy_sparse_pred}
        train_y_true_path = {"path": "datasets/Amazon-670K/train_labels.txt", "load_func": load_txt_labels}

    elif "wiki500_lightxml" in experiment:
        # WikiLarge - LightXML
        y_true_path = {"path": "datasets/Wiki-500K/test_labels.txt", "load_func": load_txt_labels}
        eta_pred_path = {"path": "predictions/wiki500/wiki500k_light_t0", "load_func": load_npy_sparse_pred}
        train_y_true_path = {"path": "datasets/Wiki-500K/train_labels.txt", "load_func": load_txt_labels}
    
    elif "wiki500_1000_lightxml" in experiment:
        # WikiLarge - LightXML
        y_true_path = {"path": "datasets/Wiki-500K/test_labels.txt", "load_func": load_txt_labels}
        eta_pred_path = {"path": "predictions/wiki500_1000/wiki500k_light_original_t0", "load_func": load_npy_sparse_pred}
        train_y_true_path = {"path": "datasets/Wiki-500K/train_labels.txt", "load_func": load_txt_labels}

    
    # Remap labels for LightXML predictions and use it when loading data
    if lightxml_data:
        with Timer():
            labels_map = calculate_lightxml_labels(train_y_true_path["path"], y_true_path["path"])
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
        eta_pred = load_cache_npz_file(**eta_pred_path)

    with Timer():
        train_y_true = load_cache_npz_file(**train_y_true_path)


    # For some sparse format this resize might be necessary
    if y_true.shape != eta_pred.shape:
        if y_true.shape[0] != eta_pred.shape[0]:
            raise RuntimeError(f"Number of instances in true and prediction do not match {y_true.shape[0]} != {eta_pred.shape[0]}")
        if y_true.shape[1] != eta_pred.shape[1]:
            new_size = max(y_true.shape[1], eta_pred.shape[1])
            eta_pred.resize((eta_pred.shape[0], new_size))
            y_true.resize((y_true.shape[0], new_size))


    # Calculate marginals and propensities
    with Timer():
        print("Calculating marginals and propensities")
        marginals = labels_priors(train_y_true)
        #marginals = labels_priors(y_true)
        inv_ps = jpv_inverse_propensity(train_y_true)


    if "apply_sigmoid" in eta_pred_path and eta_pred_path["apply_sigmoid"]:
        # LightXML predictions aren't probabilities
        eta_pred.data = 1.0 / (1.0 + np.exp(-eta_pred.data))


    # Use true labels as predictions with 1.0 score (probability)
    if true_as_pred:
        eta_pred = y_true

    print(f"y_true: type={type(y_true)}, shape={y_true.shape}")
    print(f"eta_pred: type={type(eta_pred)}, shape={eta_pred.shape}")    

    # Convert to array to check if it gives the same results
    # y_true = y_true.toarray()
    # eta_pred = eta_pred.toarray()

    # This does not work at the moment
    # y_true = y_true.todense() # As np.matrix
    # eta_pred = eta_pred.todense() # As np.matrix


    output_path_prefix = f"results/{experiment}/"
    os.makedirs(output_path_prefix, exist_ok=True)
    for k in K:
        for method, func in METHODS.items():
            print(f"{method} @ {k}: ")

            output_path = f"{output_path_prefix}{method}_k={k}_s={seed}"
            if ONLY_RESULTS:
                # if os.path.exists(output_path + "_pred_iter_1.npz"):
                #     y_pred = load_npz_wrapper(output_path + "_pred_iter_1.npz")
                #     results = report_metrics(y_true, y_pred, k)
                #     save_json(f"{output_path_prefix}{method}-iter-1_k={k}_s={seed}_results.json", results) 

                y_pred = load_npz_wrapper(output_path + "_pred.npz")
            else:
                with Timer():
                    #y_pred = func[0](eta_pred, k, marginals=marginals, inv_ps=inv_ps, filename=output_path, **func[1])
                    y_pred = func[0](eta_pred, k, marginals=marginals, inv_ps=inv_ps, seed=seed, **func[1])
                save_npz_wrapper(output_path + "_pred.npz", y_pred)
            
            print("  Calculating metrics:")
            results = report_metrics(y_true, y_pred, k)
            save_json(output_path + "_results.json", results)


if __name__ == "__main__":
    main()
