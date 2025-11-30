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
                    if i == 0 and write_flags == "w":
                        file.write(f"\n\n{e}:\n")
                        line = (
                            "| "
                            + " | ".join(
                                [f"{'method':<50}"] + [f"{k:<10}" for k in keys]
                            )
                            + " |"
                        )
                        file.write(f"{line}\n")
                        line = (
                            f"|:{'-' * 50}-|-"
                            + ":|-".join([f"{'-' * 10}" for k in keys])
                            + ":|"
                        )
                        file.write(f"{line}\n")

                    line = (
                        "| "
                        + " | ".join(
                            [f"{name:<50}"]
                            + [f"{method_results.get(k, '-'):<10}" for k in keys]
                        )
                        + " |"
                    )
                    file.write(f"{line}\n")

                elif output_format == "txt":
                    # Write header
                    if e_i == 0 and i == 0 and write_flags == "w":
                        line = "".join(
                            [f"{esc('method'):<60}"] + [f"{esc(k):<20}" for k in keys]
                        )
                        file.write(f"{line}\n")

                    line = "".join(
                        [f"{esc(name):<60}"]
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
K = (0, 1, 3)
SEEDS = (1, 23, 2024)
# SEEDS = (13)
for k in K:
    COLUMNS.extend(
        [
            # f"iP@{k}",
            # f"iR@{k}",
            # f"mP@{k}",
            # f"mR@{k}",
            f"mF@{k}",
            # f"mC@{k}",
            # f"time@{k}",
        ]
    )


experiments = [
    "results_online2/youtube_deepwalk_plt",
    "results_online2/mediamill_plt",
    "results_online2/flicker_deepwalk_plt",
    "results_online2/eurlex_plt",
    "results_online2/eurlex_lexglue_plt",
    # "results_online2/amazoncat_plt",
    # "results_online2/wiki10_plt",
]

main_methods = {
    "/default-prediction": "Top-k or >0.5",
    "/online-gd-macro-f1": "Online GD (only on test)",
    "/online-gd-macro-f1-x2": "Online GD, 2 epochs (only on test)",
    # "/online-gd-macro-f1-x5": "Online GD, 5 epochs (only on test)",
    "/online-greedy-macro-f1": "Online Greedy (only on test)",
    "/online-frank-wolfe-macro-f1": "Online FW (only on test)",
    "/online-thresholds-macro-f1": "Online find threshold (only on test)",
    "/frank-wolfe-macro-f1": "Frank-Wolfe on train",
    "/frank-wolfe-macro-f1-on-test": "Frank-Wolfe on test",
    "/frank-wolfe-macro-f1-etu": "Frank-Wolfe ETU (using only pred. marginals)",
    "/block-coord-macro-f1": "BCA (ETU, using only pred. marginals)",
    "/greedy-macro-f1": "Greedy (ETU, using only pred. marginals)",
    "/find-thresholds-macro-f1": "Find thresholds on train",
    "/find-thresholds-macro-f1-on-test": "Find thresholds on test",
    "/find-thresholds-macro-f1-etu": "Find thresholds ETU (using only pred. marginals)",
}

output_results(
    "results_md/online_methods",
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


exit(0)

experiments = [
    "results_new_mixed/mediamill_plt",
    "results_new_mixed/youtube_deepwalk_plt",
    "results_new_mixed/flicker_deepwalk_plt",
    "results_new_mixed/eurlex_plt",
    "results_new_mixed/eurlex_lexglue_plt",
    "results_new_mixed/amazoncat_plt",
    "results_new_mixed/wiki10_plt",
]
main_methods = {
    "/optimal-instance-prec": "Top-k",
    # "/optimal-instance-ps-prec": "PS-Top-k",
    # "/power-law-with-beta=0.5": "Pow-Top-k",
    # "/log": "Log-Top-k",
    # "/optimal-macro-recall": "Inv-P-Top-k",
    # ---
    "/online-block-coord-macro-f1": "10-Online-BC+BC-warmup",
    "/online-greedy-block-coord-macro-f1": "Greedy+BC-warmup",
    "/single-online-block-coord-macro-f1": "1-Online-BC+BC-warmup",
    "/single-online-block-coord-on-true-macro-f1": "1-Online-BC+BC-warmup-on-true",
    "/single-online-block-coord-with-true-c-macro-f1": "1-Online-BC+BC-warmup-true-C",
    "/online-block-coord-with-fw-macro-f1": "10-Online-BC+FW-warmup",
    "/online-greedy-block-coord-with-fw-macro-f1": "Greedy+FW-warmup",
    "/single-online-block-coord-with-fw-macro-f1": "1-Online-BC+FW-warmup",
    # ---
    "/frank-wolfe-macro-f1": "FW",
    "/frank-wolfe-average-macro-f1": "FW-avg",
    "/frank-wolfe-last-macro-f1": "FW-last",
    "/block-coord-macro-f1": "BC",
    "/greedy-macro-f1": "Greedy",
}

output_results(
    "results_md/new_methods",
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
