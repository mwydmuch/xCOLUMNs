#!/usr/bin/env bash
DATASET=(news20 ledgar sensorless cal101 cal256) # rcv1x amazoncat)
SEEDS=(1 13 23 2024 7700)

for d in "${DATASET[@]}"; do
    echo "Running python3 create_sklearn_dataset.py $d "
    python3 create_sklearn_dataset.py $d lr
done
