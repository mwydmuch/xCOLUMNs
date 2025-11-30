#!/usr/bin/env bash

set -e

EXPERIMENTS=(eurlex_100_plt EURLex-4.3K_100_plt wiki10_100_plt amazonCat_100_plt amazon_100_plt rcv1x_100_plt wikiLSHTC_100_plt WikipediaLarge-500K_100_plt)
SEEDS=(13)
KS=(1 3 5)
OUTPUT_DIR=results_thesis_bf_star
ARGS="-v 0.0 --shuffle_data --multiply_data 5 --use_proba_as_true -i 20"
NXC_ARGS="--loadAs map --ensemble 1 --threads 8 --endRow 100000"

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        echo "python3 run_thesis_2024_experiment.py $e -k $k --results_dir $OUTPUT_DIR $ARGS"
        python3 run_thesis_2024_experiment.py $e -m "bf-star" -k $k -s 13 --results_dir $OUTPUT_DIR $ARGS

        cd results_thesis_bf_star/${e}
        bash ../../nxc_predict_bf_star.sh $(echo $e | cut -d '_' -f1) . --topK 100 ${NXC_ARGS}
        bash ../../nxc_predict_bf_star.sh $(echo $e | cut -d '_' -f1) . --topK ${k} ${NXC_ARGS}
        for file in *k=${k}*.json; do
            echo $file

            if grep -q "mixed-precision-macro-precision" "$file"; then
                continue
            fi

            python3 ../../extract_bf_star_weights.py $file
            WEIGHTS_FILE=$(echo $file | sed 's/.json/_weights.txt/')
            BIASES_FILE=$(echo $file | sed 's/.json/_biases.txt/')
            WB_ARGS=""
            if [[ -e $WEIGHTS_FILE ]]; then
                wc -l $WEIGHTS_FILE
                WB_ARGS="--labelWeights ${WEIGHTS_FILE}"
            fi
            if [[ -e $BIASES_FILE ]]; then
                wc -l $BIASES_FILE
                WB_ARGS="${WB_ARGS} --labelBiases ${BIASES_FILE}"
            fi
            bash ../../nxc_predict_bf_star.sh $(echo $e | cut -d '_' -f1) . --topK ${k} ${NXC_ARGS} ${WB_ARGS}
        done
        cd ../..
    done
done

exit 0

EXPERIMENTS=(amazonCat-14K_100_plt)
SEEDS=(13)
KS=(1 3 5)
OUTPUT_DIR=results_thesis_bf_star
ARGS="-v 0.0 --shuffle_data --multiply_data 2 --use_proba_as_true -i 20"

for e in "${EXPERIMENTS[@]}"; do
    for k in "${KS[@]}"; do
        echo "python3 run_thesis_2024_experiment.py $e -k $k --results_dir $OUTPUT_DIR $ARGS"
        python3 run_thesis_2024_experiment.py $e -m "bf-star" -k $k -s 13 --results_dir $OUTPUT_DIR $ARGS

        cd results_thesis_bf_star/${e}
        bash ../../nxc_predict_bf_star.sh $(echo $e | cut -d '_' -f1) . --topK 100 ${NXC_ARGS}
        bash ../../nxc_predict_bf_star.sh $(echo $e | cut -d '_' -f1) . --topK ${k} ${NXC_ARGS}
        for file in *k=${k}*.json; do
            echo $file
            python3 ../../extract_bf_star_weights.py $file
            WEIGHTS_FILE=$(echo $file | sed 's/.json/_weights.txt/')
            BIASES_FILE=$(echo $file | sed 's/.json/_biases.txt/')
            WB_ARGS=""
            if [[ -e $WEIGHTS_FILE ]]; then
                wc -l $WEIGHTS_FILE
                WB_ARGS="--labelWeights ${WEIGHTS_FILE}"
            fi
            if [[ -e $BIASES_FILE ]]; then
                wc -l $BIASES_FILE
                WB_ARGS="${WB_ARGS} --labelBiases ${BIASES_FILE}"
            fi
            bash ../../nxc_predict_bf_star.sh $(echo $e | cut -d '_' -f1) . --topK ${k} ${NXC_ARGS} ${WB_ARGS}
        done
        cd ../..
    done
done
