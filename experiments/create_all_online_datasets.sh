#!/usr/bin/env bash
DATASET=(youtube_deepwalk mediamill flicker_deepwalk eurlex_lexglue) # rcv1x amazoncat)
#SEEDS=(1 13 23 2024 7700)
#SEEDS=(13)
SEEDS=(1 23 2024 7700)

for d in "${DATASET[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running python3 create_online_dataset.py $d linear_adam -s $seed"
        python3 create_online_dataset.py $d linear_adam -s $seed
    done
done
