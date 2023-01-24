# This file generates the results used in the tables in the paper
import functools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from prediction import *
from metrics import *
from data import load_npy_dataset


def eval_metrics(data, predictions):
    # TODO add PSP and DCG
    return {
        "mC": macro_abandonment(data, predictions),
        "mP": macro_precision(data, predictions),
        "mR": macro_recall(data, predictions),
        "iC": instance_abandonment(data, predictions),
        "iP": instance_precision(data, predictions),
        "iR": instance_recall(data, predictions)
    }


class InferenceExperiment:
    def __init__(self, dataset: str, k_values):
        self.eta = np.load(f"results/{dataset}_full_plain-scores.npy").astype(np.float32)
        self.eta = np.exp(self.eta) / (1 + np.exp(self.eta))
        self.data = load_npy_dataset(f"results/{dataset}_full_plain-labels.npy",
                                     num_ins=self.eta.shape[0],
                                     num_lbl=self.eta.shape[1])

        # estimated marginal
        self.marginal = np.mean(self.data, axis=0)

        self.k_values = k_values
        self.results = []
        self._cache = set()
        self.results_file = Path(f"results/{dataset}-inference.json")
        if self.results_file.exists():
            results = json.loads(self.results_file.read_text())
            for result in results:
                self.add_result(result)
        self.inference_functions = {}

    def register_inference_function(self, key: str, function: callable, *,
                                    group: str = "",
                                    use_true_labels: bool = False, **kwargs):
        self.inference_functions[key] = {
            "proc": functools.partial(function, **kwargs),
            "on_true": use_true_labels,
            "group": group
        }

    def run_inference(self):
        for k in self.k_values:
            self.run_inference_at_k(k)

    def add_result(self, result):
        self.results.append(result)
        self._cache.add(f"{result['Method']}@{result['k']}")

    def run_inference_at_k(self, k):
        print(f"Running at {k}")
        for key, proc in self.inference_functions.items():
            if f"{key}@{k}" in self._cache:
                continue
            print(f"Running inference for {key}")

            result = {}

            if proc["on_true"]:
                predictions = proc["proc"](self.data, k)
            else:
                predictions = proc["proc"](self.eta, k)

            metrics = eval_metrics(self.data, predictions)
            result["Method"] = key
            result["group"] = proc["group"]
            result["k"] = k
            result["true_labels"] = proc["on_true"]
            result["metrics"] = {}
            for m, v in metrics.items():
                result["metrics"][m] = float(v)

            self.add_result(result)

            self.save()

    def save(self):
        self.results_file.write_text(json.dumps(self.results, indent=2))

    def write_table(self):
        all_metrics = set()
        all_methods = set()
        tf_results = defaultdict(dict)
        for result in self.results:
            method = result["Method"]
            all_methods.add(method)
            for m, v in result["metrics"].items():
                metric = f"{m}@{result['k']}"
                all_metrics.add(metric)
                tf_results[method][metric] = v

        all_metrics = sorted(all_metrics)
        print(all_metrics)
        all_methods = sorted(all_methods)

        def esc(x):
            return "{" + x + "}"

        # write table head
        table = f"{esc('method'):<15}"
        for metric in all_metrics:
            table += f"{esc(metric):<12}"

        table += "\n"

        for method in all_methods:
            table += f"{esc(method):<15}"

            for metric in all_metrics:
                value = tf_results[method][metric]
                table += f"{value:<12.4f}"
            table += "\n"

        return table


exp = InferenceExperiment("amazoncat13k", (1, 3, 5, 10))

exp.register_inference_function("TopK", optimal_instance_precision)
exp.register_inference_function("TopK-max", optimal_instance_precision, use_true_labels=True)
exp.register_inference_function("BCP", block_coordinate_ascent_fast)
#exp.register_inference_function("BCP-max", block_coordinate_ascent_fast, use_true_labels=True)
#exp.register_inference_function("BCPo", block_coordinate_ascent_fast, mode="online")
#exp.register_inference_function("BCPo-max", block_coordinate_ascent_fast, mode="online", use_true_labels=True)
exp.register_inference_function("mR", optimal_macro_recall, marginal=exp.marginal)
#exp.register_inference_function("mR-max", optimal_macro_recall, marginal=exp.marginal, use_true_labels=True)
exp.register_inference_function("LWI", log_weighted_instance, marginal=exp.marginal)
exp.register_inference_function("SWI", sqrt_weighted_instance, marginal=exp.marginal)
#exp.register_inference_function("GCov-max", greedy_coverage, use_true_labels=True)
exp.register_inference_function("GPre", greedy_precision)
#exp.register_inference_function("BCC", block_coordinate_coverage)
#exp.register_inference_function("BCC-max", block_coordinate_coverage, use_true_labels=True)

for beta in (0.9, 0.66, 0.5, 0.33, 0.25, 0.1):
    exp.register_inference_function(f"PL-{beta}", power_law_weighted_instance, marginal=exp.marginal, beta=beta,
                                    group="power-law")

for ema in (1.0, 0.999, 0.995, 0.99, 0.95, 0.9):
    # exp.register_inference_function(f"GCov-{ema}", greedy_coverage, decay=ema, group="gcov")
    pass

exp.run_inference()
print(exp.write_table())
np.save("inference-results.npy", exp.results)
