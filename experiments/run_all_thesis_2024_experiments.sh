#!/usr/bin/env bash

# Default experiments
EXPERIMENTS=(eurlex_lightxml wiki10_lightxml amazoncat_lightxml amazon_lightxml wiki500_lightxml amazon_1000_lightxml wiki500_1000_lightxml)
EXPERIMENTS=(eurlex_lightxml wiki10_100_lightxml amazoncat_100_lightxml amazon_100_lightxml wiki500_100_lightxml)
EXPERIMENTS=(eurlex_lightxml wiki10_100_lightxml amazoncat_100_lightxml)
EXPERIMENTS=(eurlex_100_plt wiki10_100_plt amazonCat_100_plt)
EXPERIMENTS=(rcv1x_100_plt eurlex_100_plt wiki10_100_plt amazonCat_100_plt deliciousLarge_100_plt amazon_100_plt wikiLSHTC_100_plt WikipediaLarge-500K_100_plt amazonCat-14K_100_plt)
EXPERIMENTS=(rcv1x_100_plt eurlex_100_plt wiki10_100_plt amazonCat_100_plt deliciousLarge_100_plt amazon_100_plt wikiLSHTC_100_plt)
#EXPERIMENTS=(eurlex_100_plt)

# True as pred experiments
#EXPERIMENTS=(eurlex_lightxml_true_as_pred wiki10_lightxml_true_as_pred amazoncat_lightxml_true_as_pred amazon_1000_lightxml_true_as_pred wiki500_1000_lightxml_true_as_pred)

SEEDS=(13 1988 1993 2023 2024)
SEEDS=(13 1988 1993)
SEEDS=(13 1988 1993)
#SEEDS=(13)
#KS=(1 3 5 10)
KS=(3 5)

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running python3 run_thesis_2024_experiment.py $e -k $k -s $seed -v 0.0 --results_dir results_thesis_syn2 --test_multiply 3 --use_proba_as_true"
            #python3 run_thesis_2024_experiment.py $e -k $k -s $seed -v 0.0 -m frank-wolfe --results_dir results_thesis_syn3 --test_multiply 10 --use_proba_as_true &
            python3 run_thesis_2024_experiment.py $e -k $k -s $seed -v 0.0 -m frank-wolfe --results_dir results_thesis3 &
        done
        wait
    done
done

#python3 run_thesis_2024_experiment.py deliciousLarge_100_plt -k 3 -s 13 -v 0.0
