#!/usr/bin/env bash

# Different experiment sets
#EXPERIMENTS=(youtube_deepwalk_plt mediamill_plt flicker_deepwalk_plt eurlex_lexglue_plt amazoncat_plt rcv1x_plt)
#EXPERIMENTS=(news20_hsm cal101_hsm cal256_hsm sensorless_hsm protein_hsm aloi.bin_hsm ledgar_hsm)
#EXPERIMENTS=(ledgar_lr news20_lr cal101_lr cal256_lr)
EXPERIMENTS=(youtube_deepwalk_online mediamill_online flicker_deepwalk_online eurlex_lexglue_online)

KS=(0 3)
SEEDS=(1 13 23)

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running python3 run_icml_2024_omma_experiment.py $e -k $k -s $seed"
            python3 run_icml_2024_omma_experiment.py $e -k $k -s $seed -r results_online_replicate_new_reg_0 &
        done
    done
    wait
done
