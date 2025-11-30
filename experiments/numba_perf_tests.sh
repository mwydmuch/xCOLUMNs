#!/usr/bin/env bash

set -e

export NUMBA_NUM_THREADS=8
export XCOLUMNS_NUMBA_PARALLEL=1

# Default experiments
OUTPUT_DIR=results_perf_with_numba_parallel
ARGS="-v 0.0 --shuffle_data --multiply_data 5 --use_proba_as_true -i 20"
time python3 run_thesis_2024_experiment.py WikipediaLarge-500K_100_plt -m "power" -k 5 -s 13 --results_dir $OUTPUT_DIR $ARGS

export XCOLUMNS_NUMBA_PARALLEL=0

OUTPUT_DIR=results_perf_without_numba_parallel
ARGS="-v 0.0 --shuffle_data --multiply_data 5 --use_proba_as_true -i 20"
time python3 run_thesis_2024_experiment.py WikipediaLarge-500K_100_plt -m "power" -k 5 -s 13 --results_dir $OUTPUT_DIR $ARGS
