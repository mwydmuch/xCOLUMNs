#!/usr/bin/env bash
EXPERIMENTS=(eurlex wiki10 amazonCat deliciousLarge amazon wikiLSHTC rcv1x WikipediaLarge-500K amazon-3M amazonCat-14K)

# for e in "${EXPERIMENTS[@]}"; do
#     bash nxc_train_and_predict.sh ${e} "-m plt --loss l2 --maxLeaves 200 -c 1 --ensemble 3 --seed 13" "--topK 100 --loadAs sparse --ensOnTheTrot 1 --ensMissingScores 0"
# done

for e in "${EXPERIMENTS[@]}"; do
    bash nxc_train_and_predict.sh ${e} "-m plt --maxLeaves 200 -c 16 --ensemble 3 --seed 13" "--topK 100 --loadAs sparse --ensOnTheTrot 1 --ensMissingScores 0"
done
