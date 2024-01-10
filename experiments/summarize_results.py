import json
import os
import sys
import time
from math import isnan, sqrt

import numpy as np


def load_json(filepath):
    with open(filepath) as file:
        return json.load(file)


def esc(x):
    return "{" + x + "}"


def output_results(
    output,
    output_format,
    experiments,
    methods,
    columns,
    ks,
    seeds,
    select_best: bool = True,
    add_std: bool = True,
    precision: int = 4,
    write_flags: str = "w",
):
    output_path = f"{output}.{output_format}"
    with open(output_path, write_flags) as file:
        print(f"Writing {output_path}...")
        for e_i, e in enumerate(experiments):
            print(f"Experiment: {e}")
            rows = []
            best_values = {}
            second_best_values = {}

            for i, (m, name) in enumerate(methods.items()):
                method_results = {}
                for k in ks:
                    for s in seeds:
                        path1 = f"{e}{m.format(k, s)}_results.json"
                        path2 = f"{e}{m}_k={k}_s={s}_results.json"
                        if os.path.exists(path1):
                            path = path1
                        elif os.path.exists(path2):
                            path = path2
                        else:
                            print(f"Missing: {path1} / {path2}")
                            continue

                        if os.path.exists(path):
                            results = load_json(path)
                            for s in ["time", "loss", "iters"]:
                                if s in results:
                                    results[f"{s}@{k}"] = results[s]

                            for r, v in results.items():
                                if r not in method_results:
                                    method_results[r] = []
                                # if "iters" in r:  # correction for incorrectly stored iters in the old results
                                #     v = v + 1
                                if (
                                    "time" not in r
                                    and "loss" not in r
                                    and "iters" not in r
                                ):
                                    if isinstance(v, (float, int)):
                                        v *= 100
                                    else:
                                        v = [x * 100 for x in v]
                                method_results[r].append(v)

                # print(method_results)

                _method_results = {}
                for r, v in method_results.items():
                    try:
                        _method_results[r] = np.mean(v)
                        if add_std:
                            if len(v) > 1:
                                _method_results[f"{r}std"] = np.std(v)
                                _method_results[f"{r}ste"] = np.std(v) / sqrt(len(v))
                            else:
                                _method_results[f"{r}std"] = 0
                                _method_results[f"{r}ste"] = 0
                        method_results = _method_results
                    except Exception:
                        print(f"Failed to compute mean and std for {r}")

                if "time" not in method_results:
                    method_results["time"] = 0
                    if add_std:
                        method_results["timestd"] = 0
                        method_results["timeste"] = 0

                # Update max
                for k, v in method_results.items():
                    if "loss" in k:
                        current_best = best_values.get(k, 99999)
                        if v < current_best:
                            best_values[k] = v
                            second_best_values[k] = current_best
                        else:
                            second_best_values[k] = min(second_best_values.get(k, 0), v)
                    else:
                        current_best = best_values.get(k, 0)
                        if v > current_best:
                            best_values[k] = v
                            second_best_values[k] = current_best
                        else:
                            second_best_values[k] = max(second_best_values.get(k, 0), v)

                rows.append((name, method_results))

            if add_std and len(columns):
                all_columns = []
                for c in columns:
                    all_columns.append(c)
                    all_columns.append(f"{c}std")
                    all_columns.append(f"{c}ste")
            else:
                all_columns = columns

            keys = (
                sorted(method_results.keys()) if len(all_columns) == 0 else all_columns
            )

            # Write results to file
            for i, (name, method_results) in enumerate(rows):
                # Format results
                for k, v in method_results.items():
                    if "log_loss" in k:
                        v_str = f"{v:.4f}"
                    else:
                        v_str = f"{v:.2f}"
                    if select_best:
                        if "iters" in k or "time" in k or "std" in k or "ste" in k:
                            method_results[k] = v_str
                        else:
                            if v == best_values[k] and output_format == "md":
                                method_results[k] = f"**{v_str}**"
                            elif v == best_values[k] and output_format == "txt":
                                method_results[k] = f"\\textbf{{{v_str}}}"
                            elif v == second_best_values[k] and output_format == "md":
                                method_results[k] = f"*{v_str}*"
                            elif v == second_best_values[k] and output_format == "txt":
                                method_results[k] = f"\\textit{{{v_str}}}"
                            else:
                                method_results[k] = v_str
                    else:
                        method_results[k] = v_str

                if output_format == "md":
                    # Write header
                    if e_i == 0 and i == 0 and write_flags == "w":
                        line = (
                            "| "
                            + " | ".join(
                                [f"{'method':<30}"] + [f"{k:<10}" for k in keys]
                            )
                            + " |"
                        )
                        file.write(f"{line}\n")
                        line = (
                            f"|:{'-' * 30}-|-"
                            + ":|-".join([f"{'-' * 10}" for k in keys])
                            + ":|"
                        )
                        file.write(f"{line}\n")

                    line = (
                        "| "
                        + " | ".join(
                            [f"{name:<30}"] + [f"{method_results[k]:<10}" for k in keys]
                        )
                        + " |"
                    )
                    file.write(f"{line}\n")

                elif output_format == "txt":
                    # Write header
                    if e_i == 0 and i == 0 and write_flags == "w":
                        line = "".join(
                            [f"{esc('method'):<50}"] + [f"{esc(k):<20}" for k in keys]
                        )
                        file.write(f"{line}\n")

                    line = "".join(
                        [f"{esc(name):<50}"]
                        + [
                            f"{'{' + method_results.get(k, '-') + '}':<20}"
                            for k in keys
                        ]
                    )
                    file.write(f"{line}\n")


PRECISION = 4
FORMAT = "md"
COLUMNS = []
SELECT_BEST = True
ADD_STD = False
os.makedirs("results_txt", exist_ok=True)

# ETU+BCA paper
K = (3, 5)
SEEDS = (13, 26, 42, 1993, 2023)
for k in K:
    COLUMNS.extend(
        [
            f"iP@{k}",
            f"iR@{k}",
            f"mP@{k}",
            f"mR@{k}",
            f"mF@{k}",
            f"mC@{k}",
            f"iters@{k}",
            f"time@{k}",
        ]
    )

experiments = [
    "results_bca3/eurlex_lightxml",
    "results_bca/wiki10_lightxml",
    "results_bca/amazoncat_lightxml",
    "results_bca/wiki500_1000_lightxml",
    "results_bca/amazon_1000_lightxml",
]
main_methods = {
    "/optimal-instance-prec": "\\inftopk",
    "/optimal-instance-ps-prec": "\\infpstopk",
    "/power-law-with-beta=0.5": "\\infpower",
    "/log": "\\inflog",
    # ---
    "/block-coord-macro-prec-tol=1e-7": "\\infbcmacp",
    # "/optimal-macro-recall": "\\infmacr",
    "/block-coord-macro-recall-tol=1e-7": "\\infbcmacr",
    "/block-coord-macro-f1-tol=1e-7": "\\infbcmacf",
    "/block-coord-cov-tol=1e-7": "\\infbccov",
}

output_results(
    "results_txt/lightxml_main_str",
    FORMAT,
    experiments,
    main_methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
)

extended_methods = {
    "/optimal-instance-prec": "\\inftopk",
    "/optimal-instance-ps-prec": "\\infpstopk",
    "/power-law-with-beta=0.25": "\\infpower$_{\\beta=1/4}$",
    "/power-law-with-beta=0.5": "\\infpower$_{\\beta=1/2}$",
    # "power-law-with-beta=0.75": "\\infpower_{\\beta=3/4}",
    "/log": "\\inflog",
    # ---
    "/greedy-macro-prec": "\\infgreedymacp",
    "/block-coord-macro-prec-tol=1e-7": "\\infbcmacp",
    "/optimal-macro-recall": "\\infmacr",
    "/block-coord-macro-recall-tol=1e-7": "\\infbcmacr",
    "/greedy-macro-f1": "\\infgreedymacf",
    "/block-coord-macro-f1-tol=1e-7": "\\infbcmacf",
    "/greedy-cov": "\\infgreedycov",
    "/block-coord-cov-tol=1e-7": "\\infbccov",
}

output_results(
    "results_txt/lightxml_extended_str",
    FORMAT,
    experiments,
    extended_methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
)

K = (3, 5)
experiments = [
    "results_bca/amazoncat_lightxml",
    "results_bca/wiki500_lightxml",
    "results_bca/wiki500_1000_lightxml",
    "results_bca/amazon_lightxml",
    "results_bca/amazon_1000_lightxml",
]
methods = {
    # "/greedy-macro-prec": "\\infgreedymacp",
    # "/block-coord-macro-prec-iter=1": "\\infbcmacp$,\\i=1$",
    "/block-coord-macro-prec-tol=1e-3": "\\infbcmacp$,\\epsilon=10^{-3}$",
    "/block-coord-macro-prec-tol=1e-5": "\\infbcmacp$,\\epsilon=10^{-5}$",
    "/block-coord-macro-prec-tol=1e-7": "\\infbcmacp$,\\epsilon=10^{-7}$",
    # "/greedy-macro-f1": "\\infgreedymacf",
    # "/block-coord-macro-f1-iter=1": "\\infbcmacf$,\\i=1$",
    "/block-coord-macro-f1-tol=1e-3": "\\infbcmacf$,\\epsilon=10^{-3}$",
    "/block-coord-macro-f1-tol=1e-5": "\\infbcmacf$,\\epsilon=10^{-5}$",
    "/block-coord-macro-f1-tol=1e-7": "\\infbcmacf$,\\epsilon=10^{-7}$",
    # "/greedy-cov": "\\infgreedycov",
    # "/block-coord-cov-iter=1": "\\infbccov$,\\i=1$",
    "/block-coord-cov-tol=1e-3": "\\infbccov$,\\epsilon=10^{-3}$",
    "/block-coord-cov-tol=1e-5": "\\infbccov$,\\epsilon=10^{-5}$",
    "/block-coord-cov-tol=1e-7": "\\infbccov$,\\epsilon=10^{-7}$",
}

output_results(
    "results_txt/lightxml_params_str",
    FORMAT,
    experiments,
    methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
)


# PU+FW paper
ADD_STD = False
SEEDS = (21,)
experiments = [
    "results_fw/mediamill",
    "results_fw/flicker_deepwalk",
    "results_fw/rcv1x",
]
K = (3, 5, 10)
COLUMNS = []
for k in K:
    COLUMNS.extend([f"iP@{k}", f"iR@{k}", f"mP@{k}", f"mR@{k}", f"mF@{k}"])
methods = {
    "_pytorch_bce/frank-wolfe-macro-prec-rnd_k={}_s={}_t=0.0_r=0.0": "\\inffwmacp$_{\\text{-top-k}}$",
    "_pytorch_bce/frank-wolfe-macro-prec_k={}_s={}_t=0.0_r=0.0": "\\inffwmacp$_{\\text{-rnd}}$",
    "_pytorch_bce/frank-wolfe-macro-recall-rnd_k={}_s={}_t=0.0_r=0.0": "\\inffwmacr$_{\\text{-top-k}}$",
    "_pytorch_bce/frank-wolfe-macro-recall_k={}_s={}_t=0.0_r=0.0": "\\inffwmacr$_{\\text{-rnd}}$",
    "_pytorch_bce/frank-wolfe-macro-f1_k={}_s={}_t=0.0_r=0.0": "\\inffwmacf$_{\\text{-top-k}}$",
    "_pytorch_bce/frank-wolfe-macro-f1-rnd_k={}_s={}_t=0.0_r=0.0": "\\inffwmacf$_{\\text{-rnd}}$",
}

# output_results("results_txt/fw-topk-vs-random-str", FORMAT, experiments, methods, COLUMNS, K, SEEDS, SELECT_BEST, ADD_STD, PRECISION)

experiments = ["results_fw/amazoncat"]
SEEDS = (13,)
methods = {
    "_plt/frank-wolfe-macro-prec-rnd_k={}_s={}_t=0.0_r=0.0": "\\inffwmacp$_{\\text{-top-k}}$",
    "_plt/frank-wolfe-macro-prec_k={}_s={}_t=0.0_r=0.0": "\\inffwmacp$_{\\text{-rnd}}$",
    "_plt/frank-wolfe-macro-recall-rnd_k={}_s={}_t=0.0_r=0.0": "\\inffwmacr$_{\\text{-top-k}}$",
    "_plt/frank-wolfe-macro-recall_k={}_s={}_t=0.0_r=0.0": "\\inffwmacr$_{\\text{-rnd}}$",
    "_plt/frank-wolfe-macro-f1_k={}_s={}_t=0.0_r=0.0": "\\inffwmacf$_{\\text{-top-k}}$",
    "_plt/frank-wolfe-macro-f1-rnd_k={}_s={}_t=0.0_r=0.0": "\\inffwmacf$_{\\text{-rnd}}$",
}

# output_results("results_txt/fw-topk-vs-random-str", FORMAT, experiments, methods, COLUMNS, K, SEEDS, SELECT_BEST, ADD_STD, PRECISION, write_flags="a")


SEEDS = (21,)
experiments = [
    "results_fw/mediamill",
    "results_fw/flicker_deepwalk",
    "results_fw/rcv1x",
]
methods = {
    "_pytorch_bce/frank-wolfe-macro-prec_k={}_s={}_t=0.5_r=0.0": "\\inffwmacp$_{\\text{-50/50}}$",
    "_pytorch_bce/frank-wolfe-macro-prec_k={}_s={}_t=0.25_r=0.0": "\\inffwmacp$_{\\text{-75/25}}$",
    "_pytorch_bce/frank-wolfe-macro-prec_k={}_s={}_t=0.0_r=0.0": "\\inffwmacp$_{\\text{-100/100}}$",
    "_pytorch_bce/frank-wolfe-macro-recall_k={}_s={}_t=0.5_r=0.0": "\\inffwmacr$_{\\text{-50/50}}$",
    "_pytorch_bce/frank-wolfe-macro-recall_k={}_s={}_t=0.25_r=0.0": "\\inffwmacr$_{\\text{-75/25}}$",
    "_pytorch_bce/frank-wolfe-macro-recall_k={}_s={}_t=0.0_r=0.0": "\\inffwmacr$_{\\text{-100/100}}$",
    "_pytorch_bce/frank-wolfe-macro-f1_k={}_s={}_t=0.5_r=0.0": "\\inffwmacf$_{\\text{-50/50}}$",
    "_pytorch_bce/frank-wolfe-macro-f1_k={}_s={}_t=0.25_r=0.0": "\\inffwmacf$_{\\text{-75/25}}$",
    "_pytorch_bce/frank-wolfe-macro-f1_k={}_s={}_t=0.0_r=0.0": "\\inffwmacf$_{\\text{-100/100}}$",
}

# output_results("results_txt/fw-splits-str", FORMAT, experiments, methods, COLUMNS, K, SEEDS, SELECT_BEST, ADD_STD, PRECISION)


# Only balanced accuracy
os.makedirs("results_md", exist_ok=True)

ADD_STD = False
FORMAT = "md"
K = (3, 5, 10)
COLUMNS = []
for k in K:
    COLUMNS.extend([f"mBA@{k}"])
experiments = [
    "results_fw/mediamill",
    "results_fw/flicker_deepwalk",
    "results_fw/rcv1x",
]
methods = {
    "_pytorch_bce/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K",
    "_pytorch_bce/fw-split-power-law-with-beta=0.5_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{POW}}$",
    "_pytorch_bce/fw-split-log_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{LOG}}$",
    "_pytorch_focal/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K + $\\ell_{\\text{FOCAL}}$",
    "_pytorch_asym/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K + $\\ell_{\\text{ASYM}}$",
    "_pytorch_bce/fw-split-optimal-macro-balanced-acc_k={}_s={}_t=0.0_r=0.0": "Macro-BA$_{PRIOR}$",
    "_pytorch_bce/frank-wolfe-macro-balanced-acc_k={}_s={}_t=0.0_r=0.0": "Macro-BA$_{FW}$",
}

output_results(
    "results_md/fw-balance-acc",
    FORMAT,
    experiments,
    methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
)

SEEDS = (13,)
experiments = ["results_fw/amazoncat"]
methods = {
    "_plt/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K",
    "_plt/fw-split-power-law-with-beta=0.5_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{POW}}$",
    "_plt/fw-split-log_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{LOG}}$",
    "_plt/fw-split-optimal-macro-balanced-acc_k={}_s={}_t=0.0_r=0.0": "Macro-BA$_{PRIOR}$",
    "_plt/frank-wolfe-macro-balanced-acc_k={}_s={}_t=0.0_r=0.0": "Macro-BA$_{FW}$",
}

output_results(
    "results_md/fw-balance-acc",
    FORMAT,
    experiments,
    methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
    write_flags="a",
)


SEEDS = (21,)
K = (3, 5, 10)
COLUMNS = []
for k in K:
    COLUMNS.extend([f"time@{k}", f"iters@{k}"])
experiments = [
    "results_fw/mediamill",
    "results_fw/flicker_deepwalk",
    "results_fw/rcv1x",
]
methods = {
    "_pytorch_bce/frank-wolfe-macro-prec_k={}_s={}_t=0.0_r=0.0": "Macro-P$_{FW}$",
    "_pytorch_bce/frank-wolfe-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{FW}$",
    "_pytorch_bce/frank-wolfe-macro-f1_k={}_s={}_t=0.0_r=0.0": "Macro-F1$_{FW}$",
}

output_results(
    "results_md/fw-time-iter",
    FORMAT,
    experiments,
    methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
)

SEEDS = (13,)
experiments = ["results_fw/amazoncat"]
methods = {
    "_plt/frank-wolfe-macro-prec_k={}_s={}_t=0.0_r=0.0": "Macro-P$_{FW}$",
    "_plt/frank-wolfe-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{FW}$",
    "_plt/frank-wolfe-macro-f1_k={}_s={}_t=0.0_r=0.0": "Macro-F1$_{FW}$",
}

output_results(
    "results_md/fw-time-iter",
    FORMAT,
    experiments,
    methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
    write_flags="a",
)


SEEDS = (21,)
K = (5,)
COLUMNS = []
for k in K:
    COLUMNS.extend([f"iP@{k}", f"iR@{k}", f"mP@{k}", f"mR@{k}", f"mF@{k}"])
experiments = [
    "results_fw_final/mediamill",
    "results_fw_final/flicker_deepwalk",
    "results_fw_final/rcv1x",
]
methods = {
    "_pytorch_bce/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K",
    "_pytorch_bce/fw-split-power-law-with-beta=0.5_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{POW}}$",
    "_pytorch_bce/fw-split-log_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{LOG}}$",
    "_pytorch_focal/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K + $\\ell_{\\text{FOCAL}}$",
    "_pytorch_asym/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K + $\\ell_{\\text{ASYM}}$",
    "_pytorch_bce/frank-wolfe-macro-prec_k={}_s={}_t=0.0_r=0.0": "Macro-P$_{FW}$",
    "_pytorch_bce/fw-split-bca-macro-prec_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{BCA}$",
    "_pytorch_bce/fw-split-optimal-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{PRIOR}$",
    "_pytorch_bce/frank-wolfe-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{FW}$",
    "_pytorch_bce/fw-split-bca-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{BCA}$",
    "_pytorch_bce/frank-wolfe-macro-f1_k={}_s={}_t=0.0_r=0.0": "Macro-F1$_{FW}$",
    "_pytorch_bce/fw-split-bca-macro-f1_k={}_s={}_t=0.0_r=0.0": "Macro-F1$_{BCA}$",
}

output_results(
    "results_md/fw-vs-bca",
    FORMAT,
    experiments,
    methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
)

SEEDS = (13,)
experiments = ["results_fw_final/amazoncat"]
methods = {
    "_plt/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K",
    "_plt/fw-split-power-law-with-beta=0.5_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{POW}}$",
    "_plt/fw-split-log_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{LOG}}$",
    "_plt/frank-wolfe-macro-prec_k={}_s={}_t=0.0_r=0.0": "Macro-P$_{FW}$",
    "_plt/fw-split-bca-macro-prec_k={}_s={}_t=0.0_r=0.0": "Macro-P$_{BCA}$",
    "_plt/fw-split-optimal-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{PRIOR}$",
    "_plt/frank-wolfe-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{FW}$",
    "_plt/fw-split-bca-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{BCA}$",
    "_plt/frank-wolfe-macro-f1_k={}_s={}_t=0.0_r=0.0": "Macro-F1$_{FW}$",
    "_plt/fw-split-bca-macro-f1_k={}_s={}_t=0.0_r=0.0": "Macro-F1$_{BCA}$",
}

output_results(
    "results_md/fw-vs-bca",
    FORMAT,
    experiments,
    methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
    write_flags="a",
)

SEEDS = (21,)
experiments = ["results_fw/amazoncat"]
methods = {
    "_plt/fw-split-optimal-instance-prec_k={}_s={}_t=0.0_r=0.0": "Top-K",
    "_plt/fw-split-power-law-with-beta=0.5_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{POW}}$",
    "_plt/fw-split-log_k={}_s={}_t=0.0_r=0.0": "Top-K + $w^{\\text{LOG}}$",
    "_plt/frank-wolfe-macro-prec_k={}_s={}_t=0.0_r=0.0": "Macro-P$_{FW}$",
    "_plt/fw-split-bca-macro-prec_k={}_s={}_t=0.0_r=0.0": "Macro-P$_{BCA}$",
    "_plt/fw-split-optimal-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{PRIOR}$",
    "_plt/frank-wolfe-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{FW}$",
    "_plt/fw-split-bca-macro-recall_k={}_s={}_t=0.0_r=0.0": "Macro-R$_{BCA}$",
    "_plt/frank-wolfe-macro-f1_k={}_s={}_t=0.0_r=0.0": "Macro-F1$_{FW}$",
    "_plt/fw-split-bca-macro-f1_k={}_s={}_t=0.0_r=0.0": "Macro-F1$_{BCA}$",
}

output_results(
    "results_md/fw-vs-bca",
    FORMAT,
    experiments,
    methods,
    COLUMNS,
    K,
    SEEDS,
    SELECT_BEST,
    ADD_STD,
    PRECISION,
    write_flags="a",
)
