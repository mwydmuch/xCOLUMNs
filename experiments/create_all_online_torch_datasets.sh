#!/usr/bin/env bash
DATASET=(news20 youtube_deepwalk mediamill flicker_deepwalk eurlex_lexglue) # rcv1x amazoncat)
SEEDS=(1 13 23 2024 7700)
DATASET=(eurlex_lexglue)

SEEDS=(2024 7700)

for d in "${DATASET[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running python3 create_online_torch_dataset.py $d linear_adam -s $seed"
        python3 create_online_torch_dataset.py $d linear_adam -s $seed
    done
done
