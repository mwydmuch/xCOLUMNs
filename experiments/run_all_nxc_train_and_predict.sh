#!/usr/bin/env bash


EXPERIMENTS=(eurlex wiki10 amazonCat deliciousLarge amazon wikiLSHTC rcv1x amazon-3M EURLex-4.3K)
for e in "${EXPERIMENTS[@]}"; do
    bash nxc_train_and_predict.sh ${e} "-m plt --maxLeaves 200 -c 16 --ensemble 3 --seed 13" "--topK 100 --loadAs map --ensOnTheTrot 1 --ensMissingScores 0 --threads 8"
    bash nxc_train_and_predict.sh ${e} "-m plt --maxLeaves 400 -c 32 --ensemble 3 --seed 13" "--topK 100 --loadAs map --ensOnTheTrot 1 --ensMissingScores 0 --threads 8"
done

EXPERIMENTS=(WikipediaLarge-500K amazonCat-14K)
for e in "${EXPERIMENTS[@]}"; do
    bash nxc_train_and_predict.sh ${e} "-m plt --maxLeaves 200 -c 16 --ensemble 3 --seed 13" "--topK 100 --loadAs map --ensOnTheTrot 1 --ensMissingScores 0 --batchRows 400000 --threads 8"
    bash nxc_train_and_predict.sh ${e} "-m plt --maxLeaves 400 -c 32 --ensemble 3 --seed 13" "--topK 100 --loadAs map --ensOnTheTrot 1 --ensMissingScores 0 --batchRows 400000 --threads 8"
done
