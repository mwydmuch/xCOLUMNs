#!/usr/bin/env bash

set -e
set -o pipefail

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
ROOT_DIR=napkinXC

DATASET_NAME=$1
MODEL_DIR=predictions/nxc

# If there are exactly 3 arguments and 2 starts with nxc parameter (-)
if [[ $# -gt 2 ]] && [[ $2 == -* ]] && [[ $3 == -* ]]; then
    TRAIN_ARGS=$2
    TEST_ARGS=$3
    if [[ $# -gt 3 ]]; then
        MODEL_DIR=$4
    fi
else
    shift
    TRAIN_ARGS="$@"
    TEST_ARGS=""
fi

TRAIN_CONFIG=${DATASET_NAME}_$(echo "${TRAIN_ARGS}" | tr " /" "__")
TEST_CONFIG=${TRAIN_CONFIG}_$(echo "${TEST_ARGS}" | tr " /" "__")

MODEL=${MODEL_DIR}/${TRAIN_CONFIG}
DATASET_DIR=datasets/${DATASET_NAME}
DATASET_FILE=${DATASET_DIR}/${DATASET_NAME}

# Download dataset
if [[ ! -e $DATASET_DIR ]]; then
    python3 ${ROOT_DIR}/experiments/get_dataset.py $DATASET_NAME bow-v1 datasets
fi

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

# Train model
TRAIN_RESULT_FILE=${MODEL}/train_results
TRAIN_LOCK_FILE=${MODEL}/.train_lock
if [[ ! -e $MODEL ]] || [[ -e $TRAIN_LOCK_FILE ]]; then
    mkdir -p $MODEL
    touch $TRAIN_LOCK_FILE

    ${ROOT_DIR}/nxc train -i $TRAIN_FILE -o $MODEL $TRAIN_ARGS | tee $TRAIN_RESULT_FILE
    echo "Train date: $(date)" | tee -a $TRAIN_RESULT_FILE
    echo "Model file size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+[BMG]' -o)" | tee -a $TRAIN_RESULT_FILE
    echo "Model file size (K): $(du -c ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+' -o)" | tee -a $TRAIN_RESULT_FILE
    rm -f $TRAIN_LOCK_FILE
fi

# Predict using model
RESULT_FILE=${MODEL}/train_${TEST_CONFIG}
LOCK_FILE=${MODEL}/.test_train_lock_${TEST_CONFIG}
if [[ ! -e $RESULT_FILE ]] || [[ -e $LOCK_FILE ]]; then
    touch $LOCK_FILE
    if [ -e $TRAIN_RESULT_FILE ]; then
        cat $TRAIN_RESULT_FILE > $RESULT_FILE
    fi

    if [[ $TEST_ARGS == *"--labelsWeights"* ]]; then
        TEST_ARGS="${TEST_ARGS} --labelsWeights ${INV_PS_FILE}"
    fi

    PRED_CONFIG=$(echo "${TEST_ARGS}" | tr " /" "__")
    PRED_FILE=${MODEL}/train_pred_${PRED_CONFIG}
    PRED_LOCK_FILE=${MODEL}/.test_train_lock_${PRED_CONFIG}
    PRED_RESULT_FILE=${MODEL}/train_pred_results_${PRED_CONFIG}
    if [[ ! -e $PRED_FILE ]] || [[ -e $PRED_LOCK_FILE ]]; then
        touch $PRED_LOCK_FILE
        ${ROOT_DIR}/nxc test -i $TRAIN_FILE -o $MODEL $TEST_ARGS --prediction $PRED_FILE --measures "" | tee -a $PRED_RESULT_FILE
        rm -rf $PRED_LOCK_FILE
    fi

    if [ -e $PRED_RESULT_FILE ]; then
        cat $PRED_RESULT_FILE >> $RESULT_FILE
    fi

    ln -s train_pred_${PRED_CONFIG} ${MODEL}/train_pred

    echo "Pred date: $(date)" | tee -a $RESULT_FILE
    rm -rf $LOCK_FILE
else
    cat $RESULT_FILE
fi


RESULT_FILE=${MODEL}/test_${TEST_CONFIG}
LOCK_FILE=${MODEL}/.test_test_lock_${TEST_CONFIG}
if [[ ! -e $RESULT_FILE ]] || [[ -e $LOCK_FILE ]]; then
    touch $LOCK_FILE
    if [ -e $TRAIN_RESULT_FILE ]; then
        cat $TRAIN_RESULT_FILE > $RESULT_FILE
    fi

    if [[ $TEST_ARGS == *"--labelsWeights"* ]]; then
        TEST_ARGS="${TEST_ARGS} --labelsWeights ${INV_PS_FILE}"
    fi

    PRED_CONFIG=$(echo "${TEST_ARGS}" | tr " /" "__")
    PRED_FILE=${MODEL}/test_pred_${PRED_CONFIG}
    PRED_LOCK_FILE=${MODEL}/.test_test_lock_${PRED_CONFIG}
    PRED_RESULT_FILE=${MODEL}/test_pred_results_${PRED_CONFIG}
    if [[ ! -e $PRED_FILE ]] || [[ -e $PRED_LOCK_FILE ]]; then
        touch $PRED_LOCK_FILE
        ${ROOT_DIR}/nxc test -i $TEST_FILE -o $MODEL $TEST_ARGS --prediction $PRED_FILE --measures "" | tee -a $PRED_RESULT_FILE
        rm -rf $PRED_LOCK_FILE
    fi

    if [ -e $PRED_RESULT_FILE ]; then
        cat $PRED_RESULT_FILE >> $RESULT_FILE
    fi

    if [[ -e ${MODEL}/test_pred ]]; then
        rm -f ${MODEL}/test_pred
    fi
    ln -s test_pred_${PRED_CONFIG} ${MODEL}/test_pred

    wc -l ${MODEL}/test_pred

    python3 ${ROOT_DIR}/experiments/evaluate.py $TEST_FILE $PRED_FILE $INV_PS_FILE | tee -a $RESULT_FILE

    echo "Pred date: $(date)" | tee -a $RESULT_FILE
    rm -rf $LOCK_FILE
else
    cat $RESULT_FILE
fi

# Create a symlink to the model directory
if [[ -e ${MODEL_DIR}/${DATASET_NAME} ]]; then
    rm -f ${MODEL_DIR}/${DATASET_NAME}
fi

ln -s ${TRAIN_CONFIG} ${MODEL_DIR}/${DATASET_NAME}

# Create a symlink to the model directory
# if [[ -e ${MODEL_DIR}/${DATASET_NAME}_l2 ]]; then
#     rm -f ${MODEL_DIR}/${DATASET_NAME}_l2
# fi

# ln -s ${TRAIN_CONFIG} ${MODEL_DIR}/${DATASET_NAME}_l2
