#!/usr/bin/env bash

set -e
set -o pipefail

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
ROOT_DIR=${SCRIPT_DIR}/napkinXC

DATASET_NAME=$1
PRED_DIR=$2
MODEL_DIR=${SCRIPT_DIR}/predictions/nxc

shift
shift
TEST_ARGS="$@"

TEST_CONFIG=$(echo "${TEST_ARGS}" | tr " /" "__" | awk -F"--labelBiases" '{print $1}')

MODEL=${MODEL_DIR}/${DATASET_NAME}/member_0
cp ${MODEL_DIR}/${DATASET_NAME}/args.bin ${MODEL}/args.bin
DATASET_DIR=${SCRIPT_DIR}/datasets/${DATASET_NAME}
DATASET_FILE=${DATASET_DIR}/${DATASET_NAME}

# Find train / test file
if [[ -e "${DATASET_FILE}.train.remapped" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train.remapped"
    TEST_FILE="${DATASET_FILE}.test.remapped"
elif [[ -e "${DATASET_FILE}_train.txt.remapped" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt.remapped"
    TEST_FILE="${DATASET_FILE}_test.txt.remapped"
elif [[ -e "${DATASET_FILE}_train.txt" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt"
    TEST_FILE="${DATASET_FILE}_test.txt"
elif [[ -e "${DATASET_FILE}.train" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train"
    TEST_FILE="${DATASET_FILE}.test"
elif [[ -e "${DATASET_FILE}_train.libsvm" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.libsvm"
    TEST_FILE="${DATASET_FILE}_test.libsvm"
elif [[ -e "${DATASET_FILE}_train.svm" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.svm"
    TEST_FILE="${DATASET_FILE}_test.svm"
fi

LOCK_FILE=${PRED_DIR}/.test_test_lock_${TEST_CONFIG}
if [[ ! -e $RESULT_FILE ]] || [[ -e $LOCK_FILE ]]; then
    touch $LOCK_FILE

    PRED_FILE=${PRED_DIR}/test_pred_${TEST_CONFIG}
    PRED_LOCK_FILE=${MODEL}/.test_test_lock_${TEST_CONFIG}
    PRED_RESULT_FILE=${PRED_DIR}/test_pred_results_${TEST_CONFIG}
    if [[ ! -e $PRED_FILE ]] || [[ -e $PRED_LOCK_FILE ]]; then
        touch $PRED_LOCK_FILE
        ${ROOT_DIR}/nxc test -i $TEST_FILE -o $MODEL $TEST_ARGS --prediction $PRED_FILE --measures "" | tee -a $PRED_RESULT_FILE
        rm -rf $PRED_LOCK_FILE
    fi
    python3 ${ROOT_DIR}/experiments/evaluate.py $TEST_FILE $PRED_FILE | tee -a $PRED_RESULT_FILE

    echo "Pred date: $(date)" | tee -a $PRED_RESULT_FILE
    rm -rf $LOCK_FILE
else
    cat $RESULT_FILE
fi
