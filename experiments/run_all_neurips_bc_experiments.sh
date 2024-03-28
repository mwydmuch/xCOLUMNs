#!/usr/bin/env bash

# Default experiments
EXPERIMENTS=(eurlex_lightxml wiki10_lightxml amazoncat_lightxml amazon_lightxml wiki500_lightxml amazon_1000_lightxml wiki500_1000_lightxml)

# True as pred experiments
#EXPERIMENTS=(eurlex_lightxml_true_as_pred wiki10_lightxml_true_as_pred amazoncat_lightxml_true_as_pred amazon_1000_lightxml_true_as_pred wiki500_1000_lightxml_true_as_pred)

SEEDS=(13 26 42 1993 2023)
KS=(1 3 5 10)

for e in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for k in "${KS[@]}"; do
            echo "Running python3 run_neurips_bc_experiment.py $e -k $k -s $seed"
            python3 run_neurips_bc_experiment.py $e -k $k -s $seed &
        done
        wait
    done
done
