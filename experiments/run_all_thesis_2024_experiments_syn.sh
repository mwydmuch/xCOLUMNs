#!/usr/bin/env bash

set -e

export NUMBA_NUM_THREADS=8
export XCOLUMNS_NUMBA_PARALLEL=1

# Default experiments
SEEDS=(13)
KS=(1 3 5)
OUTPUT_DIR=results_thesis_syn4
ARGS="-v 0.0 --shuffle_data --multiply_data 5 --use_proba_as_true -i 20"
#ARGS="-v 0.0 --shuffle_data --multiply_data 5 --use_proba_as_true"
EXPERIMENTS=(rcv1x_100_plt eurlex_100_plt EURLex-4.3K_100_plt amazonCat_100_plt wiki10_100_plt wikiLSHTC_100_plt WikipediaLarge-500K_100_plt amazon_100_plt)
EXPERIMENTS=(wikiLSHTC_100_plt WikipediaLarge-500K_100_plt)

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "python3 run_thesis_2024_experiment.py $e -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS"
            python3 run_thesis_2024_experiment.py $e -m "power" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "ps-precision" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "optimal" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "log" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "1e-08" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "coverage" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
        done
    done
done

ARGS="-v 0.0 --shuffle_data --multiply_data 2 --use_proba_as_true -i 20"
#ARGS="-v 0.0 --shuffle_data --multiply_data 5 --use_proba_as_true"
EXPERIMENTS=(amazonCat-14K_100_plt)

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "python3 run_thesis_2024_experiment.py $e -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS"
            python3 run_thesis_2024_experiment.py $e -m "power" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "ps-precision" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "optimal" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "log" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "1e-08" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
            python3 run_thesis_2024_experiment.py $e -m "coverage" -k $k -s $seed --results_dir $OUTPUT_DIR $ARGS
        done
    done
done
