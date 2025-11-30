#!/usr/bin/env bash

set -e

EXPERIMENTS=(rcv1x_100_plt eurlex_100_plt wiki10_100_plt amazonCat_100_plt deliciousLarge_100_plt amazon_100_plt wikiLSHTC_100_plt WikipediaLarge-500K_100_plt amazonCat-14K_100_plt)
SEEDS=(13 1988 1993 2023 2024)
KS=(1 3 5)
OUTPUT_DIR=results_thesis_org
ARGS="-v 0.0"

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "python3 run_thesis_2024_experiment.py $e -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS &"
            python3 run_thesis_2024_experiment.py $e -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS &
        done
        wait
    done
done
