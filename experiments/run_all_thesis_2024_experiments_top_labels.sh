#!/usr/bin/env bash

set -e

# Default experiments
EXPERIMENTS=(rcv1x_100_plt eurlex_100_plt EURLex-4.3K_100_plt amazonCat_100_plt amazonCat-14K_100_plt wiki10_100_plt wikiLSHTC_100_plt WikipediaLarge-500K_100_plt amazon_100_plt)

SEEDS=(13)
KS=(1 3 5)
OUTPUT_DIR=results_thesis_org
ARGS="-v 0.0 --top_labels 0.1"

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        echo "python3 run_thesis_2024_experiment.py $e -k $k --results_dir $OUTPUT_DIR $ARGS"
        python3 run_thesis_2024_experiment.py $e -m "optimal" -k $k -s 13 --results_dir $OUTPUT_DIR $ARGS
    done
done

ARGS="-v 0.0 --top_labels 0.2"

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        echo "python3 run_thesis_2024_experiment.py $e -k $k --results_dir $OUTPUT_DIR $ARGS"
        python3 run_thesis_2024_experiment.py $e -m "optimal" -k $k -s 13 --results_dir $OUTPUT_DIR $ARGS
    done
done
