#!/usr/bin/env bash

SEEDS=(21 42 63) # 147 168 189 210 231 252)
DATASETS=(mediamill flicker_deepwalk rcv1x)
KS=(3 5)
METHODS=(pytorch_bce pytorch_focal pytorch_asym)
for d in "${DATASETS[@]}"; do
    for m in "${METHODS[@]}"; do
        for k in "${KS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "Running python3 run_iclr_2024_fw_experiment.py ${d}_${m} -k $k -s $seed -t 0"
                python3 run_iclr_2024_fw_experiment.py ${d}_${m} -k $k -s $seed -t 0.0 -m "fw-split-optimal-instance-prec" &
            done
            wait
        done
    done
done

METHODS=(pytorch_bce)
for d in "${DATASETS[@]}"; do
    for m in "${METHODS[@]}"; do
        for k in "${KS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "Running python3 run_iclr_2024_fw_experiment.py ${d}_${m} -k $k -s $seed -t 0"
                python3 run_iclr_2024_fw_experiment.py ${d}_${m} -k $k -s $seed -t 0.0 &
            done
            wait
        done
    done
done

DATASETS=(amazoncat)
METHODS=(plt)
KS=(3 5 10)
for d in "${DATASETS[@]}"; do
    for m in "${METHODS[@]}"; do
        for k in "${KS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "Running python3 run_iclr_2024_fw_experiment.py ${d}_${m} -k $k -s $seed -t 0"
                python3 run_iclr_2024_fw_experiment.py ${d}_${m} -k $k -s $seed -t 0.0 &
            done
        done
        wait
    done
done
