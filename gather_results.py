import time
import json
import numpy as np
import sys
import os

datasets = ["eurlex", "wiki10", "amazoncat", "amazon"]
algos = ["plt", "lightxml"]
methods = {
    "optimal-instance-prec": "\\inftopk",
    "optimal-instance-ps-prec": "\\infpstopk",
    "power-law-with-beta=0.25": "\\infpower$_{\\beta=1/4}$",
    "power-law-with-beta=0.5": "\\infpower$_{\\beta=1/2}$",
    #"power-law-with-beta=0.75": "\\infpower_{\\beta=3/4}",
    "log": "\\inflog",
    # ---
    "block-coord-macro-prec-iter-1": "\\infbcmacp$_{i=1}$",
    "block-coord-macro-prec": "\\infbcmacp",
    "optimal-macro-recall": "\\infmacr",
    "block-coord-macro-f1-iter-1": "\\infbcmacf$_{i=1}$",
    "block-coord-macro-f1": "\\infbcmacf",
}

experiments = [f"{d}_{a}" for d in datasets for a in algos]
    
K = (1, 3, 5)
K = (3,)

SEPARATOR = ""

def load_json(filepath):
    with open(filepath) as file:
        return json.load(file)
    
def esc(x):
    return "{" + x + "}"


for e in experiments:
    output_path = f"tex/{e}.txt"
    with open(output_path, "w") as file:
        for i, (m, name) in enumerate(methods.items()):
            method_results = {}
            for k in K:
                path = f"results/{e}/{m}@{k}_results.json"
                if os.path.exists(path):
                    results = load_json(path)
                    method_results.update(results)

            # Write header
            if i == 0:
                keys = sorted(method_results.keys())
                line = SEPARATOR.join([f"{esc('method'):<50}"] + [f"{esc(k):<12}" for k in keys])
                file.write(f"{line}\n")
            
            line = SEPARATOR.join([f"{esc(name):<50}"] + [f"{method_results.get(k,0):<12.4f}" for k in keys])
            file.write(f"{line}\n")


os.system("cat tex/eurlex_lightxml.txt > tex/all_lightxml.txt")
os.system("cat tex/amazoncat_lightxml.txt | tail -n +2 >> tex/all_lightxml.txt")
os.system("cat tex/wiki10_lightxml.txt | tail -n +2 >> tex/all_lightxml.txt")
os.system("cat tex/amazon_lightxml.txt | tail -n +2 >> tex/all_lightxml.txt")

os.system("cat tex/eurlex_plt.txt > tex/all_plt.txt")
os.system("cat tex/amazoncat_plt.txt | tail -n +2 >> tex/all_plt.txt")
os.system("cat tex/wiki10_plt.txt | tail -n +2 >> tex/all_plt.txt")
os.system("cat tex/amazon_plt.txt | tail -n +2 >> tex/all_plt.txt")