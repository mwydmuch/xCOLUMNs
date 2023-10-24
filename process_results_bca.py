import time
import json
import numpy as np
import sys
import os
from math import sqrt, isnan


PRECISION = 4
FORMAT = "txt"
COLUMNS = []
SELECT_BEST = True
ADD_STD = True

# ETU papet
SPLIT=0.0
OUTPUT="tex/main_lightxml_str2"
algos = ["lightxml"]
experiments = ["eurlex_lightxml", "wiki10_lightxml", "amazoncat_lightxml", "wiki500_1000_lightxml", "amazon_1000_lightxml"]
K = (1, 3, 5, 10)
SEEDS = (13, 26, 42, 1993, 2023)
for k in K:
    COLUMNS.extend([f"iP@{k}", f"iR@{k}", f"mP@{k}", f"mR@{k}", f"mF@{k}", f"mC@{k}", f"iters@{k}", f"time@{k}"])
main_methods = {
    "/optimal-instance-prec": "\\inftopk",
    "/optimal-instance-ps-prec": "\\infpstopk",
    "/power-law-with-beta=0.5": "\\infpower",
    "/log": "\\inflog",
    # ---
    "/block-coord-macro-prec-tol=1e-7": "\\infbcmacp",
#    "/optimal-macro-recall": "\\infmacr",
    "/block-coord-macro-recall-tol=1e-7": "\\infbcmacr",
    "/block-coord-macro-f1-tol=1e-7": "\\infbcmacf",
    "/block-coord-cov-tol=1e-7": "\\infbccov",
}

detailed_methods = {
    "/optimal-instance-prec": "\\inftopk",
    "/optimal-instance-ps-prec": "\\infpstopk",
    "/power-law-with-beta=0.25": "\\infpower$_{\\beta=1/4}$",
    "/power-law-with-beta=0.5": "\\infpower$_{\\beta=1/2}$",
    #"power-law-with-beta=0.75": "\\infpower_{\\beta=3/4}",
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

methods=main_methods
#methods=detailed_methods


SELECT_BEST=True
OUTPUT="tex/params_lightxml"
experiments = ["amazoncat_lightxml", "wiki500_lightxml", "wiki500_1000_lightxml", "amazon_lightxml", "amazon_1000_lightxml"]
#experiments = ["eurlex_lightxml", "wiki10_lightxml" ,"amazoncat_lightxml", "amazon_lightxml", "amazon_1000_lightxml", "wiki500_lightxml", "wiki500_1000_lightxml"]
methods = {
    #"/greedy-macro-prec": "\\infgreedymacp",
    #"/block-coord-macro-prec-iter=1": "\\infbcmacp$,\\i=1$",
    "/block-coord-macro-prec-tol=1e-3": "\\infbcmacp$,\\epsilon=10^{-3}$",
    "/block-coord-macro-prec-tol=1e-5": "\\infbcmacp$,\\epsilon=10^{-5}$",
    "/block-coord-macro-prec-tol=1e-7": "\\infbcmacp$,\\epsilon=10^{-7}$",
    #"/greedy-macro-f1": "\\infgreedymacf",
    #"/block-coord-macro-f1-iter=1": "\\infbcmacf$,\\i=1$",
    "/block-coord-macro-f1-tol=1e-3": "\\infbcmacf$,\\epsilon=10^{-3}$",
    "/block-coord-macro-f1-tol=1e-5": "\\infbcmacf$,\\epsilon=10^{-5}$",
    "/block-coord-macro-f1-tol=1e-7": "\\infbcmacf$,\\epsilon=10^{-7}$",
    #"/greedy-cov": "\\infgreedycov",
    #"/block-coord-cov-iter=1": "\\infbccov$,\\i=1$",
    "/block-coord-cov-tol=1e-3": "\\infbccov$,\\epsilon=10^{-3}$",
    "/block-coord-cov-tol=1e-5": "\\infbccov$,\\epsilon=10^{-5}$",
    "/block-coord-cov-tol=1e-7": "\\infbccov$,\\epsilon=10^{-7}$",
}


def load_json(filepath):
    with open(filepath) as file:
        return json.load(file)
    
def esc(x):
    return "{" + x + "}"


output_path = f"{OUTPUT}.{FORMAT}"
with open(output_path, "w") as file:
    print(f"Writing {output_path}...")
    for e_i, e in enumerate(experiments):
        #output_path = f"tex/{e}.{FORMAT}"
        print(f"Experiment: {e}")
        rows = []
        best_values = {}
        second_best_values = {}
        
        for i, (m, name) in enumerate(methods.items()):
            method_results = {}
            for k in K:
                for s in SEEDS:
                    path0 = f"results_bca/{e}{m}_results.json".format(k, s)
                    path1 = f"results_bca/{e}{m}_k={k}_s={s}_results.json"
                    path2 = f"results_bca/{e}{m}_k={k}_s={s}_t={SPLIT}_r=0.0_results.json"
                    path3 = f"results_bca/{e}{m}_k={k}_s={s}_t={SPLIT}_results.json"
                    if os.path.exists(path0):
                        path = path0
                    elif os.path.exists(path3):
                        path = path3
                    elif os.path.exists(path2):
                        path = path2
                    elif os.path.exists(path1):
                        path = path1
                    else:
                        print(f"Missing: {path1}") # or {path1} or {path2}")
                        continue

                    if os.path.exists(path):
                        results = load_json(path)
                        for s in ["time", "loss", "iters"]:
                            if s in results:
                                results[f"{s}@{k}"] = results[s]

                        for r, v in results.items():
                            if r not in method_results:
                                method_results[r] = []
                            if "iters" in r:
                                v = v + 1
                            if "time" not in r and "loss" not in r and "iters" not in r:
                                if isinstance(v, (float, int)):
                                    v *= 100
                                else:
                                    v = [x * 100 for x in v]
                            method_results[r].append(v)

            _method_results = {}
            for r, v in method_results.items():
                _method_results[r] = np.mean(v)
                if ADD_STD:
                    if len(v) > 1:
                        _method_results[f"{r}std"] = np.std(v)
                        _method_results[f"{r}ste"] = np.std(v) / sqrt(len(v))
                    else:
                        _method_results[f"{r}std"] = 0
                        _method_results[f"{r}ste"] = 0
                method_results = _method_results

            if "time" not in method_results:
                method_results["time"] = 0
                if ADD_STD:
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
        
        if ADD_STD and len(COLUMNS):
            columns = []
            for c in COLUMNS:
                columns.append(c)
                columns.append(f"{c}std")
                columns.append(f"{c}ste")
        else:
            columns = COLUMNS

        keys = sorted(method_results.keys()) if len(columns) == 0 else columns
        
        # Write results to file
        for i, (name, method_results) in enumerate(rows):
            # Format results
            for k, v in method_results.items():
                if "log_loss" in k:
                    v_str = f"{v:.4f}"
                else:
                    v_str = f"{v:.2f}"
                if SELECT_BEST:
                    if "iters" in k or "time" in k or "std" in k or "ste" in k:
                        method_results[k] = v_str
                    else:
                        if v == best_values[k] and FORMAT == "md":
                            method_results[k] = f"**{v_str}**"
                        elif v == best_values[k] and  FORMAT == "txt":
                            method_results[k] = f"\\textbf{{{v_str}}}"
                        elif v == second_best_values[k] and FORMAT == "md":
                            method_results[k] = f"*{v_str}*"
                        elif v == second_best_values[k] and FORMAT == "txt":
                            method_results[k] = f"\\textit{{{v_str}}}"
                        else:
                            method_results[k] = v_str
                else:
                    method_results[k] = v_str

            if FORMAT == "md":
                # Write header
                if e_i == 0 and i == 0:
                    line = "| " + " | ".join([f"{'method':<20}"] + [f"{k:<10}" for k in keys]) + " |"
                    file.write(f"{line}\n")
                    line = f"|:{'-' * 20}-|-" + ":|-".join([f"{'-' * 10}" for k in keys]) + ":|"
                    file.write(f"{line}\n")
                
                line = "| " + " | ".join([f"{name:<20}"] + [f"{method_results[k]:<10}" for k in keys]) + " |"
                file.write(f"{line}\n")


            elif FORMAT == "txt":
                # Write header
                if e_i == 0 and i == 0:
                    line = "".join([f"{esc('method'):<50}"] + [f"{esc(k):<20}" for k in keys])
                    file.write(f"{line}\n")
                
                line = "".join([f"{esc(name):<50}"] + [f"{'{' + method_results.get(k, '-') + '}':<20}" for k in keys])
                file.write(f"{line}\n")
