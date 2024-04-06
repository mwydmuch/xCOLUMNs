#!/usr/bin/env bash

EXPERIMENTS=(youtube_deepwalk_plt mediamill_plt flicker_deepwalk_plt eurlex_lexglue_plt amazoncat_plt rcv1x_plt)
EXPERIMENTS=(youtube_deepwalk_plt mediamill_plt flicker_deepwalk_plt eurlex_lexglue_plt)
KS=(0 3)
SEEDS=(1 13 23 2024 7700)

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running python3 run_online_experiment.py $e -k $k -s $seed"
            python3 run_online_experiment.py $e -k $k -s $seed &
        done
        wait
    done
done
